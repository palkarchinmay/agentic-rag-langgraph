"""Users may only index into / query their own collection."""
import pytest
from fastapi import HTTPException
from api.middleware.auth import own_collection, assert_collection_access

ALICE = {"sub": "uid1", "email": "alice@example.com", "role": "user"}
ADMIN = {"sub": "uid9", "email": "root@example.com", "role": "admin"}


def test_own_collection_uses_email():
    assert own_collection(ALICE) == "user_alice@example.com"


def test_user_can_use_own_and_default():
    assert_collection_access(ALICE, "user_alice@example.com")
    assert_collection_access(ALICE, "default")


def test_user_cannot_use_someone_elses():
    with pytest.raises(HTTPException) as e:
        assert_collection_access(ALICE, "user_bob@example.com")
    assert e.value.status_code == 403


def test_admin_can_use_any():
    assert_collection_access(ADMIN, "user_bob@example.com")


def test_email_case_does_not_matter():
    assert_collection_access(ALICE, "user_Alice@Example.com")
