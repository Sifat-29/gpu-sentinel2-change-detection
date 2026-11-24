# lib/

Third-party code that is not installed through the system package manager.

| File | Version | Licence | Purpose |
| --- | --- | --- | --- |
| `stb_image_write.h` | v1.19, commit `6b5799bae8a1fe3792691ef4b72ae26434c3e799` of https://github.com/nothings/stb | Public domain (Unlicense) or MIT, your choice, as stated at the end of the file | PNG writer for `change_map.png` and quick-diff images |

SHA-256 of the vendored file:
`5d89edc8bffd848f3e138b2a1dbec67aadbed591894bde16ade7c83dc233af9b`.

The file is used unmodified. It is excluded from `make lint` and `make format`
(only `src/` and `tests/` are checked). Include it from exactly one source file
with `#define STB_IMAGE_WRITE_IMPLEMENTATION`.
