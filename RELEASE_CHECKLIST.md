# Release Checklist

Use this before publishing a new package release.

## Packaging

- Verify `pyproject.toml` version is bumped.
- Confirm optional extras still match the documented install commands.
- Build the wheel and sdist locally.
- Inspect the built artifacts for the expected package name and metadata.

## Validation

- Run the core test suite.
- Run the integration tests that cover optional adapters.
- Import `gistlattice` without installing any extras.
- Import each extra path after installing the matching dependency set.

## Documentation

- Update `README.md` if the install surface changed.
- Update example instructions if a new optional dependency was added.
- Check any compatibility tables or option matrices.

## Publish

- Tag the release in git.
- Upload the artifacts to PyPI.
- Verify the installed package from a clean environment.
