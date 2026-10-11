# Changelog

## 0.7.1 (2026-04-19)

### Bug Fixes

- **ci**: drop removed sr action force input ([ce1ad4d](https://github.com/urmzd/languide/commit/ce1ad4d3ffa64e20be2faa37405f233a11ea9fdf))

### Misc

- migrate sr config and action to v7 ([a127571](https://github.com/urmzd/languide/commit/a127571c4e9af551cfcdfd2ec197e8f4373c9e1f))
- migrate sr config and action to v4 ([25370ea](https://github.com/urmzd/languide/commit/25370ea1590d68f56b03e7f4c5f66159dc0fad76))
- add linguist overrides to fix language stats ([c711b2b](https://github.com/urmzd/languide/commit/c711b2bf157b1855aa8842ad95fc74168420fe61))
- **deps**: bump actions/create-github-app-token from 1 to 3 ([346bd2f](https://github.com/urmzd/languide/commit/346bd2fdf7e5d296bc58529863fb09c8d902f61f))

[Full Changelog](https://github.com/urmzd/languide/compare/v0.7.0...v0.7.1)


## 0.7.0 (2026-04-04)

### Features

- add teasr demo config (#11) ([03994cb](https://github.com/urmzd/languide/commit/03994cb23f1e2379cbe162b16b8e66c9fba19dad))

[Full Changelog](https://github.com/urmzd/languide/compare/v0.6.1...v0.7.0)


## 0.6.1 (2026-03-30)

### Documentation

- **skills**: migrate to agentskills.io spec structure ([36740de](https://github.com/urmzd/languide/commit/36740de439a1ce90f92f6598ca26f697281e0652))
- apply README standards with centered header and structured sections ([aebc250](https://github.com/urmzd/languide/commit/aebc2509a1c902c8e5635273fc5a08ef8e72c63b))

### Refactoring

- replace Python CLI with shell scripts, unify build pipeline ([15bf85e](https://github.com/urmzd/languide/commit/15bf85ef3c10a5df3a0399e99e1a17d2ec5e68c4))
- replace Python CLI with shell script ([0e27f2e](https://github.com/urmzd/languide/commit/0e27f2ed11f8cf56fe1c22d895afc5f73bf51602))
- unify build pipeline and eliminate duplicated discovery logic ([bbcbfcb](https://github.com/urmzd/languide/commit/bbcbfcbd73c9652dd6dbe35dc38f701063dc0db0))

### Miscellaneous

- update sr action from v2 to v3 ([6913aa9](https://github.com/urmzd/languide/commit/6913aa9201613627f1517e8980f6a4b2d5cafbaa))
- **scripts**: use pre-increment operators in build-guide.sh ([9aa9616](https://github.com/urmzd/languide/commit/9aa961669497817b4649dccff5484d32b61fe99d))
- add git hooks for sr integration ([475d5cc](https://github.com/urmzd/languide/commit/475d5cc8f757d6eaea4fdbeede9bf22dcea58c13))
- split into ci.yml and cd.yml, remove run-guide.yml ([2140e14](https://github.com/urmzd/languide/commit/2140e1456fd5af46db27b8cbeab4ccdefd0dad42))
- standardize CI/CD — add sr.yaml, ci.yml, release workflow, justfile ([62601dd](https://github.com/urmzd/languide/commit/62601dde7a997cb3b74716ac1c11b1acd1d84b02))
- update semantic-release action to sr@v2 ([6bf1b90](https://github.com/urmzd/languide/commit/6bf1b9049e023bf23800e03153adad44705cc52b))

[Full Changelog](https://github.com/urmzd/languide/compare/v0.6.0...v0.6.1)


## 0.6.0 (2026-03-17)

### Features

- add AGENTS.md, create-language skill, and rewrite README ([bc55699](https://github.com/urmzd/languide/commit/bc55699712a647538a5f0cd3d4b90a00b95d0d64))

### Miscellaneous

- standardize GitHub Actions workflows ([dc6d0ce](https://github.com/urmzd/languide/commit/dc6d0ce1174fd6f3570917c937b73167607bcfa0))
- add sensitive paths to .gitignore ([4ace631](https://github.com/urmzd/languide/commit/4ace631ce5aa2f3af391402fa72243b5b464d0a8))
- update semantic-release action to v1 ([e26ba76](https://github.com/urmzd/languide/commit/e26ba7630084b8c4d36f82168337418c86e29525))
- license under Apache 2.0 ([907e94c](https://github.com/urmzd/languide/commit/907e94c8a92930b3f6c7445e9c1a38b8e22184c7))

[Full Changelog](https://github.com/urmzd/languide/compare/v0.5.0...v0.6.0)


## 0.5.0 (2026-02-11)

### Features

- add force re-release support to release workflow ([3ec4435](https://github.com/urmzd/languide/commit/3ec443503ea520a6a693b8fe743bfb9fef893ea8))

### Documentation

- add note about PDF artifacts in releases ([9acca7a](https://github.com/urmzd/languide/commit/9acca7aa0119a1f98d55dc906d6b3eb2b5b7bc7c))
- rename project to Languide in README to match repo rename ([f665baf](https://github.com/urmzd/languide/commit/f665baf8bc2712fe2a656c094ebce92d423baa29))


## 0.4.0 (2026-02-08)

### Features

- use semantic-release built-in artifact upload instead of separate step ([1a44362](https://github.com/urmzd/languide/commit/1a443626ba44359cdedddf41cde91a0a99b8e3d2))

### Contributors

- @urmzd


## 0.3.0 (2026-02-08)

### Features

- build before release and attach PDFs as release artifacts ([d02855e](https://github.com/urmzd/languide/commit/d02855e145ef97cc9d1110587a1f7b6615eed64a))

### Contributors

- @urmzd


## 0.2.0 (2026-02-08)

### Features

- add Spanish guide, fix CJK conditional loading, and clean up project ([4ba76ba](https://github.com/urmzd/languide/commit/4ba76ba2808f0c175248087c3be707e7f6cd7bbf))

### Contributors

- @urmzd


## 0.1.0 (2026-02-08)

### Features

- add semantic release and simplify CI/CD to single-job pipeline ([2867dd8](https://github.com/urmzd/languide/commit/2867dd834fb56d8a56d255a3e3f12b701e5d4e98))

### Bug Fixes

- use chapters ([9e059d7](https://github.com/urmzd/languide/commit/9e059d72dd9f500c09fd48d93e230eb6fefceccd))
- extend formatting ([ff4081a](https://github.com/urmzd/languide/commit/ff4081a96bdbbbe32a51b5cfe545692c7f413f7e))

### Contributors

- @urmzd
