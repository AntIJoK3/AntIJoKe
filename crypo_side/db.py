from contextlib import contextmanager
from typing import Generator

from neo4j import Driver, GraphDatabase

from crypo_side.config import get_settings


def make_driver() -> Driver:
    s = get_settings()
    return GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))


@contextmanager
def session_scope() -> Generator:
    driver = make_driver()
    try:
        with driver.session() as session:
            yield session
    finally:
        driver.close()
