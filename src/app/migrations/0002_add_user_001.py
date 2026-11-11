"""add user 001

Revision ID: 0002add001
Revises: 0001create
"""
import os

import sqlalchemy as sa
from alembic import op

revision = "0002add001"
down_revision = "0001create"
branch_labels = None
depends_on = None

USER_ID = "001"
USERNAME = "001"


def upgrade() -> None:
    # Random scrypt-format hash nobody knows: the row exists but cannot log in until a password is set.
    locked_hash = f"scrypt${os.urandom(16).hex()}${os.urandom(64).hex()}"
    op.execute(
        sa.text("INSERT INTO users (id, username, password_hash, created) VALUES (:id, :username, :hash, strftime('%s','now'))")
        .bindparams(id=USER_ID, username=USERNAME, hash=locked_hash)
    )


def downgrade() -> None:
    op.execute(sa.text("DELETE FROM tokens WHERE user_id = :id").bindparams(id=USER_ID))
    op.execute(sa.text("DELETE FROM threads WHERE user_id = :id").bindparams(id=USER_ID))
    op.execute(sa.text("DELETE FROM users WHERE id = :id").bindparams(id=USER_ID))
