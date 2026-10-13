# Making a new release of jupyter_mynerva

## Python package

Configure a [Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
for this repository, using workflow `release.yml` and environment `pypi`.
Releases from forks use TestPyPI and environment `testpypi`.

The required version of nblibram must be available on PyPI before a production
release of Mynerva.

1. Update the version in `package.json` and the release notes in `CHANGELOG.md`.
2. Commit the changes and push a matching version tag, such as `v<version>`.
3. The Release workflow builds the packages, tests the wheel, and attaches the
   tested packages to a draft GitHub Release.
4. Review and publish the draft Release. This uploads its Python packages to
   PyPI without rebuilding them.

## NPM package

The GitHub Release also includes the frontend package. To publish it to npm,
download the `.tgz` asset and run:

```bash
npm login
npm publish ./jupyter-mynerva-<version>.tgz --access public
```
