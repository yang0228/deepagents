from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import modal
import pytest
from deepagents.backends.protocol import FileDownloadResponse, FileUploadResponse

from langchain_modal import ModalSandbox

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (PermissionError("read denied"), "permission_denied"),
        (IsADirectoryError("directory"), "is_directory"),
        (FileNotFoundError("missing"), "file_not_found"),
    ],
)
def test_download_continues_after_file_error(
    tmp_path: Path, error: OSError, expected: str
) -> None:
    valid = tmp_path / "valid.txt"
    valid.write_bytes(b"downloaded")
    sandbox = Mock(spec=modal.Sandbox)
    with valid.open("rb") as handle:
        sandbox.open.side_effect = [error, handle]
        backend = ModalSandbox(sandbox=sandbox)
        responses = backend.download_files(["/blocked", "/valid.txt"])

    assert responses == [
        FileDownloadResponse(path="/blocked", content=None, error=expected),
        FileDownloadResponse(path="/valid.txt", content=b"downloaded", error=None),
    ]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (PermissionError("write denied"), "permission_denied"),
        (IsADirectoryError("directory"), "is_directory"),
        (FileNotFoundError("missing"), "file_not_found"),
    ],
)
def test_upload_continues_after_file_error(
    tmp_path: Path, error: OSError, expected: str
) -> None:
    valid = tmp_path / "valid.txt"
    sandbox = Mock(spec=modal.Sandbox)
    with valid.open("wb") as handle:
        sandbox.open.side_effect = [error, handle]
        backend = ModalSandbox(sandbox=sandbox)
        responses = backend.upload_files(
            [("/blocked", b"blocked"), ("/valid.txt", b"uploaded")]
        )

    assert responses == [
        FileUploadResponse(path="/blocked", error=expected),
        FileUploadResponse(path="/valid.txt", error=None),
    ]
    assert valid.read_bytes() == b"uploaded"
