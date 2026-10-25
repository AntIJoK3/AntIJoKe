from __future__ import annotations

import asyncio
import hashlib
import re
import socket
import ssl
from dataclasses import dataclass
from urllib.parse import urlparse

import dns.resolver
import httpx
import tldextract
from bs4 import BeautifulSoup

try:
    import whois
except ImportError:  # pragma: no cover
    whois = None  # type: ignore


# 用 tldextract 内置的 PSL 快照：不联网、不写 .suffix_cache，
# 也避免并发请求争抢同一个文件锁（否则 /predict 会抛 "file lock could not be acquired"）。
# 5.3.x 里 suffix_list_urls=() 仍会走缓存文件锁，因此直接加载随包自带的快照。
_tld = tldextract.TLDExtract(
    suffix_list_urls=(),
    cache_dir=None,
    fallback_to_snapshot=True,
)


UA = (
    "Mozilla/5.0 (compatible; FraudGraphBot/0.1; +https://example.local) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@dataclass
class EnrichmentResult:
    url: str
    domain: str
    title: str
    visible_text: str
    ips: list[str]
    ns_names: list[str]
    registrant_emails: list[str]
    analytics_ids: list[tuple[str, str]]  # (type, id)
    cert_fingerprint: str | None
    cert_subject_cn: str | None
    content_hash: str
    raw_error: str | None = None


def _normalize_url(url: str) -> str:
    u = url.strip()
    if not u.startswith(("http://", "https://")):
        u = "http://" + u
    return u


def _extract_domain(url: str) -> str:
    p = urlparse(url)
    host = p.hostname or ""
    ext = _tld(host)
    if ext.registered_domain:
        return ext.registered_domain
    return host.lower()


def _dns_a(domain: str, lifetime: float = 5.0) -> list[str]:
    out: list[str] = []
    try:
        for a in dns.resolver.resolve(domain, "A", lifetime=lifetime):
            out.append(str(a))
    except Exception:
        pass
    try:
        for a in dns.resolver.resolve(domain, "AAAA", lifetime=lifetime):
            out.append(str(a))
    except Exception:
        pass
    return sorted(set(out))


def _dns_ns(domain: str, lifetime: float = 5.0) -> list[str]:
    out: list[str] = []
    try:
        for r in dns.resolver.resolve(domain, "NS", lifetime=lifetime):
            out.append(str(r.target).rstrip(".").lower())
    except Exception:
        pass
    return sorted(set(out))


def _whois_emails(domain: str) -> list[str]:
    if whois is None:
        return []
    try:
        w = whois.whois(domain)
    except Exception:
        return []
    emails: set[str] = set()
    for key in ("emails", "email"):
        val = getattr(w, key, None)
        if isinstance(val, str):
            emails.add(val.lower())
        elif isinstance(val, list):
            for e in val:
                if isinstance(e, str):
                    emails.add(e.lower())
    return sorted(emails)


_GA_RE = re.compile(
    r"(?:google-analytics\.com/(?:analytics\.js|ga\.js)|googletagmanager\.com/gtag/js\?id=)(?:[^\"'\s]*)",
    re.I,
)
_UA_RE = re.compile(r"UA-\d{4,10}-\d{1,4}", re.I)
_GA4_RE = re.compile(r"G-[A-Z0-9]{6,12}", re.I)
_ADSENSE_RE = re.compile(r"ca-pub-\d{10,20}", re.I)


def _extract_analytics(html: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for m in _UA_RE.findall(html):
        found.append(("UA", m))
    for m in _GA4_RE.findall(html):
        found.append(("GA4", m))
    for m in _ADSENSE_RE.findall(html):
        found.append(("ADSENSE", m))
    for m in _GA_RE.findall(html):
        if "id=" in m:
            found.append(("GTM_SNIPPET", m[:120]))
    return list(dict.fromkeys(found))


def _cert_meta(domain: str, port: int = 443, timeout: float = 5.0) -> tuple[str | None, str | None]:
    ctx = ssl.create_default_context()
    try:
        with socket.create_connection((domain, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=domain) as ssock:
                cert_bin = ssock.getpeercert(binary_form=True)
                if not cert_bin:
                    return None, None
                digest = hashlib.sha256(cert_bin).hexdigest()
                cert_dict = ssock.getpeercert()
                subj = dict(x[0] for x in cert_dict.get("subject", ()))
                cn = subj.get("commonName")
                return digest, cn
    except Exception:
        return None, None


async def _fetch_http(
    client: httpx.AsyncClient,
    url: str,
    timeout: float,
) -> tuple[str, str, str, str | None]:
    """返回 title, visible_text, html, error。"""
    title = ""
    text = ""
    html = ""
    err: str | None = None
    try:
        r = await client.get(url, timeout=timeout)
        r.raise_for_status()
        html = r.text
        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())[:8000]
    except Exception as e:  # noqa: BLE001
        err = str(e)
    return title, text, html, err


async def _dns_parallel(domain: str, lifetime: float) -> tuple[list[str], list[str]]:
    ips, ns = await asyncio.gather(
        asyncio.to_thread(_dns_a, domain, lifetime),
        asyncio.to_thread(_dns_ns, domain, lifetime),
    )
    return ips, ns


async def _whois_bounded(domain: str, timeout: float) -> list[str]:
    if timeout <= 0 or whois is None:
        return []
    try:
        return await asyncio.wait_for(asyncio.to_thread(_whois_emails, domain), timeout=timeout)
    except (asyncio.TimeoutError, Exception):
        return []


async def _cert_bounded(domain: str, timeout: float) -> tuple[str | None, str | None]:
    if not domain or timeout <= 0:
        return None, None
    try:
        return await asyncio.wait_for(asyncio.to_thread(_cert_meta, domain, 443, timeout), timeout=timeout)
    except (asyncio.TimeoutError, Exception):
        return None, None


async def enrich_url_async(
    url: str,
    timeout: float = 5.0,
    *,
    dns_lifetime: float = 2.0,
    whois_timeout: float = 2.0,
    cert_timeout: float = 2.0,
) -> EnrichmentResult:
    """并行执行 HTTP / DNS / WHOIS / TLS 证书探测。"""
    norm = _normalize_url(url)
    domain = _extract_domain(norm)

    async with httpx.AsyncClient(
        headers={"User-Agent": UA},
        follow_redirects=True,
    ) as client:
        http_task = asyncio.create_task(_fetch_http(client, norm, timeout))
        dns_task = asyncio.create_task(_dns_parallel(domain, dns_lifetime))
        whois_task = asyncio.create_task(_whois_bounded(domain, whois_timeout))
        cert_task = asyncio.create_task(_cert_bounded(domain, cert_timeout))

        http_res, dns_res, emails, cert_res = await asyncio.gather(
            http_task, dns_task, whois_task, cert_task
        )

    title, text, html, err = http_res
    ips, ns_names = dns_res
    fp, cn = cert_res

    content_hash = hashlib.sha256((html or norm).encode("utf-8", errors="ignore")).hexdigest()
    analytics = _extract_analytics(html or "")

    return EnrichmentResult(
        url=norm,
        domain=domain,
        title=title,
        visible_text=text,
        ips=ips,
        ns_names=ns_names,
        registrant_emails=emails,
        analytics_ids=analytics,
        cert_fingerprint=fp,
        cert_subject_cn=cn,
        content_hash=content_hash,
        raw_error=err,
    )


def enrich_url(
    url: str,
    timeout: float = 5.0,
    *,
    dns_lifetime: float = 2.0,
    whois_timeout: float = 2.0,
    cert_timeout: float = 2.0,
) -> EnrichmentResult:
    """同步入口：内部使用 asyncio 并行丰富化（供 /predict、导入脚本调用）。"""
    coro = enrich_url_async(
        url,
        timeout=timeout,
        dns_lifetime=dns_lifetime,
        whois_timeout=whois_timeout,
        cert_timeout=cert_timeout,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # 已在事件循环内（如未来 async 路由）：在线程中跑独立 loop
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(
            asyncio.run,
            enrich_url_async(
                url,
                timeout=timeout,
                dns_lifetime=dns_lifetime,
                whois_timeout=whois_timeout,
                cert_timeout=cert_timeout,
            ),
        ).result()
