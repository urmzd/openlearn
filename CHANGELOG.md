# Changelog

## 0.12.0 (2026-10-11)

### Features

- **course**: reuse verdicts in check --all and cache mutation results across runs ([41550de](https://github.com/urmzd/openlearn/commit/41550de9bf5e09bcd8f7de12a1e0c9f99949c3ed))
- **course**: make L8.3 and load.02 optional side modules ([2c04ad9](https://github.com/urmzd/openlearn/commit/2c04ad93631cdf084845969ccd30fa642aa150fa))
- **course**: reference durable and worker entry points (dep.06) ([448bdc3](https://github.com/urmzd/openlearn/commit/448bdc31fe214c720993df4ccd73d969e3983252))
- **course**: give build modules real call sites ([113651f](https://github.com/urmzd/openlearn/commit/113651f7636e8fbae0cb21efbd44156b7e80fce0))
- **course**: apply shared contract and harness items from open-items ([71d4385](https://github.com/urmzd/openlearn/commit/71d43856a18de08aa6cb06695ca015b92f0dc543))
- **course**: finish the no-FFI restructure and align DESIGN.md ([d49911f](https://github.com/urmzd/openlearn/commit/d49911ff0c0dea48ce62149799660505ce7c9328))
- **course**: checkpoint Codex step B progress (work in progress) ([012a495](https://github.com/urmzd/openlearn/commit/012a495eb84456c3a932d60afc66e2ec454f1ed6))
- **course**: complete Step B process boundaries and optional C ([a897e20](https://github.com/urmzd/openlearn/commit/a897e2066fc72a0cf65568efe9cb55f161a2b8a8))
- **course**: complete Step A module artifacts ([cec5587](https://github.com/urmzd/openlearn/commit/cec5587ed81c1688d8779986c0bd1a25acbfa131))
- **course**: checkpoint Codex step A progress (work in progress) ([60d4f8a](https://github.com/urmzd/openlearn/commit/60d4f8a0910b781e5525cdbcbe9e00968de85d58))
- **course**: checkpoint B7-4 through B10-3, B14 design, and handoff ([ba209bf](https://github.com/urmzd/openlearn/commit/ba209bf74bc6b70e49a5180ce34b0c06dd0cb869))
- **course**: checkpoint module groups B2 through B7-3 (work in progress) ([7cfc027](https://github.com/urmzd/openlearn/commit/7cfc027b4e3c24b3d62ac0c7d4a0877bd4be6117))
- **course**: checkpoint foundation and first module groups (work in progress) ([d54ddea](https://github.com/urmzd/openlearn/commit/d54ddea6dc34df7795707a90ec5ac2f40516078b))
- **course**: verify and harden B1 tracer batch ([1340572](https://github.com/urmzd/openlearn/commit/1340572d5556aa81aa3cb1c26e08c0b03ef21170))
- **course**: B1 harness, contracts, and tracer modules (work in progress) ([f52bdac](https://github.com/urmzd/openlearn/commit/f52bdac2e57c0fab125f0635884bb1b66709a70a))

### Bug Fixes

- **course**: serve a 1024-position copy of tiny-llama-2l so tool prompts fit ([ef066dd](https://github.com/urmzd/openlearn/commit/ef066dd92b3b6c9ac30435f60abd39eff173d56d))
- **course**: under --ci a sampled mutation estimate below the bar falls back to the full grade ([fe5a958](https://github.com/urmzd/openlearn/commit/fe5a958816d7da06af5b93d83c9225524cd765ab))
- **course**: green per-module verify for Go, ops, and gateway modules ([38ac89b](https://github.com/urmzd/openlearn/commit/38ac89bde117a949fc0b23b2a5e45d5cabd2d62c))
- **course**: batch-invariant candle runner, platform-stable mutants, e2e commit guard ([2e946e1](https://github.com/urmzd/openlearn/commit/2e946e1b34d63d013e316cc65d345d2ef05fd65f))
- **course**: MS-L5 smoke trains 2000 updates so the EM bar holds on Linux ([c71f3bd](https://github.com/urmzd/openlearn/commit/c71f3bda122f469f7f1dee6059e26b10459931b5))
- **course**: regenerate mutant patches against marker-free reference text ([d1d1b3a](https://github.com/urmzd/openlearn/commit/d1d1b3ab5773cb07c7a28608ce2ef79c3f71835d))
- **course**: sample mutation grades for the whole e2e run ([44719a7](https://github.com/urmzd/openlearn/commit/44719a7c7a873acec72f3a63fb6a3854f004a3f5))
- **course**: regenerate L8.3 mutants, sample e2e mutation grades ([920d3f0](https://github.com/urmzd/openlearn/commit/920d3f0710b036407155f3284d72aaa61ca3c95f))
- **course**: move the C lowp mutants from M09.4 to the optional M09.7 ([52bc794](https://github.com/urmzd/openlearn/commit/52bc79445a0892ba19a4cc0ad883fa7f95875262))
- **course**: engine spans, SIGTERM, and crate roots for the v1 reference engine ([3d7a782](https://github.com/urmzd/openlearn/commit/3d7a782b2cdcb1dfa6ef2a272c72dd937f03bef8))
- **course**: v1 engine in the reference system, parity drivers, sharded PR verify ([ca7740e](https://github.com/urmzd/openlearn/commit/ca7740e61d14466c5d6ad3b381af42b0724961d6))
- **course**: close e2e and global verify gaps ([a1548a6](https://github.com/urmzd/openlearn/commit/a1548a6a0f36ca11b517376a645f31312581ab0f))
- **course**: realign harness self-tests and C contracts with the no-FFI design ([8abecd7](https://github.com/urmzd/openlearn/commit/8abecd79ec8c75dd86a436a32c78672cf42b41f9))

### Misc

- **course**: update the handoff with the state after steps B, D, and E ([22638d5](https://github.com/urmzd/openlearn/commit/22638d5451d52e19ee24513a7a79219e9d11eaaf))
- **course**: 120-minute e2e and kind jobs; e2e names every non-passing module ([bd252d5](https://github.com/urmzd/openlearn/commit/bd252d5f574535711ef2e8c1505865a26e209e96))
- **release**: read the sr-releaser key from the release environment (#21) ([8f8dc73](https://github.com/urmzd/openlearn/commit/8f8dc7390f4993776aa2903da28ff02db426ca5b))
- **course**: update the handoff with current status and CI findings ([8a5c9c6](https://github.com/urmzd/openlearn/commit/8a5c9c625a8d24a40551006d45630410cd4a81fa))
- **course**: give the kind milestone job 60 minutes ([06e90eb](https://github.com/urmzd/openlearn/commit/06e90eb43ff1d230c7102034bfdd896ea0861df9))
- **course**: fix remaining lint and Linux-only harness test failures ([1e3fcbb](https://github.com/urmzd/openlearn/commit/1e3fcbb417125145ec8a50ace212bf0b879c1df2))
- **course**: fix course verify, kind deploy, and e2e failures on Linux ([ddb0cd3](https://github.com/urmzd/openlearn/commit/ddb0cd376ff2aea1cb006a9bc95738a10ca328ea))
- **course**: fix lint on the course tree ([842a438](https://github.com/urmzd/openlearn/commit/842a438f8316a5a22b6338ad454aa66fc9e42aed))
- upgrade urmzd/sr action to v9 (#20) ([7b54c8d](https://github.com/urmzd/openlearn/commit/7b54c8d5b5ce9bbcf86b100d73ca613203055153))
- **course**: add end-to-end course design (draft v2) ([fbe0cae](https://github.com/urmzd/openlearn/commit/fbe0cae8025847f45abf5e6ee7ad073cd7d9e3cb))

[Full Changelog](https://github.com/urmzd/openlearn/compare/v0.11.0...v0.12.0)


## 0.11.0 (2026-10-09)

### Features

- **site**: generate llms.txt via starlight-llms-txt ([8aa2f4a](https://github.com/urmzd/supersource/commit/8aa2f4a7f9ada4baa9ae285076e88fa5c436466e))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.10.0...v0.11.0)


## 0.10.0 (2026-10-09)

### Features

- **site**: add Starlight website generated from the curriculum markdown ([43eeff9](https://github.com/urmzd/supersource/commit/43eeff9979652389eba33c676eebc981e8a7a385))
- **paths**: split the FDE path into six parts composed by Superstar FDE ([204b148](https://github.com/urmzd/supersource/commit/204b1486b39c877de4dbb1cff138c60805ec2f50))
- **practice**: let paths include other paths in ss learn ([e3ec3a1](https://github.com/urmzd/supersource/commit/e3ec3a1315fe71718786984829547696268c672d))
- **paths**: add the FDE inference role path and wire it into the curriculum ([1e80e60](https://github.com/urmzd/supersource/commit/1e80e602b453651ca9ddfe2d61ed84eb1bdfe1f9))
- **practice**: add ss learn to read role paths stage by stage ([18d6d9d](https://github.com/urmzd/supersource/commit/18d6d9dc09a04c4fb5eadfcb99021d5255ac7b7b))
- **practice**: pin the JDK with mise and resolve it inside ss ([a9e95c8](https://github.com/urmzd/supersource/commit/a9e95c84d5608d38e280e31b26a8e002bde64d8b))
- **practice**: complete the C++ track with five more exercises ([90b81eb](https://github.com/urmzd/supersource/commit/90b81ebfb99fae342edbe57f4defd4eebd5a3d49))
- **practice**: add five C++ build exercises ([041d3a1](https://github.com/urmzd/supersource/commit/041d3a133a14ca106dbdf7c2b90149d46e24fc26))
- **practice**: add ten verified C build exercises ([dae41a4](https://github.com/urmzd/supersource/commit/dae41a4a15ac967b71bef8db20cde46e6106a56f))
- **practice**: assert wall time and peak memory per snippet ([37eb131](https://github.com/urmzd/supersource/commit/37eb1318e8b20aae60ee63d7c17ea215e24a5410))
- **practice**: fill the predict harness with 24 verified snippets ([a63ab7b](https://github.com/urmzd/supersource/commit/a63ab7bd704cfb1ec8374949c7db67d400bd8fab))
- **practice**: add predict-then-run harness baseline ([36a7a18](https://github.com/urmzd/supersource/commit/36a7a18de65d0117568ba5bc621e008be0fca24b))
- **programming-languages**: add Type Systems & Polymorphism track (#14) ([b27eb92](https://github.com/urmzd/supersource/commit/b27eb92148b3a7cefc3b979a8a09ddef952e509f))
- **data-engineering**: add runnable Streamflow orders pipeline (Spark + dbt + Airflow) ([af58da6](https://github.com/urmzd/supersource/commit/af58da68904059ebf72ec04277a7b97ac1bfb04d))
- **infrastructure**: split diagramming-and-operations into focused tracks ([1a3bf0d](https://github.com/urmzd/supersource/commit/1a3bf0d22e473cd04131041e9e25ca35af9b367d))
- **ml**: Neural Architectures & the History of Deep Learning (#8) ([3acadc5](https://github.com/urmzd/supersource/commit/3acadc5451aa04020313bd308ecf18982ac99b29))
- **swiss-table**: scaffold Swiss Table hash table implementation ([d263754](https://github.com/urmzd/supersource/commit/d263754347f26eee9744eae1300fb0efaa6acb76))
- **skills**: add study-session skill for interactive learning ([858e936](https://github.com/urmzd/supersource/commit/858e9361af9ecdfbb458e2f59e805ac4dc22188b))
- **skills**: add study-plan skill for personalized learning ([e3909c8](https://github.com/urmzd/supersource/commit/e3909c841643bf3966d430c16a537b1be80036b1))
- **skills**: add practice-impl skill for scaffolding implementations ([f56c193](https://github.com/urmzd/supersource/commit/f56c1936d7cd7527501d7fbb5176481236c75c7c))
- add mathematics curriculum foundation ([1b1b901](https://github.com/urmzd/supersource/commit/1b1b90191837de31664e1b104890dc757cd16655))
- **patterns**: add existing pattern guides for all topics ([cff4133](https://github.com/urmzd/supersource/commit/cff4133fc898d0ec56d6b7de12376796766e282f))
- **patterns**: add ML & Statistics patterns guide ([2228206](https://github.com/urmzd/supersource/commit/2228206bb0af45cc455049e87053dfe1009340fb))
- **interviews**: add comprehensive interview guides for 19 top companies ([d94a418](https://github.com/urmzd/supersource/commit/d94a418c2d643a33234ef85b9284f716c0cdbba1))
- **probabilistic-structures**: add bloom filter event validator ([25a7c87](https://github.com/urmzd/supersource/commit/25a7c8762cfb90abbde8f7ff22e6062f9860b63a))
- **ml-stats**: add linear regression implementation ([cdadf59](https://github.com/urmzd/supersource/commit/cdadf59d4e5f3fbe093743fe42b0250ea780a7e7))
- **greedy**: implement Huffman coding compression algorithm ([59bb42f](https://github.com/urmzd/supersource/commit/59bb42f5cd6749ad28fedb23f407e991ffa3ca16))
- **graphs**: add Charles and the Corgi Conundrum solution template ([4001307](https://github.com/urmzd/supersource/commit/4001307a90e902c8f8677e9ccd766fa62d97cd79))
- **practice**: add TF-IDF vector search implementation ([4a8dbfc](https://github.com/urmzd/supersource/commit/4a8dbfc9838dc0a3c99ea324dbdf3e1c9e9e588e))
- **practice**: add k-means clustering implementation ([5bcd03a](https://github.com/urmzd/supersource/commit/5bcd03a598dae28c9d4290136e39af17837fbae0))
- **main**: add entry point script ([ff9b027](https://github.com/urmzd/supersource/commit/ff9b02700e8f6932ceb4ba0d62768bd424f4891f))

### Bug Fixes

- **practice**: include stdint.h and memory explicitly so gcc builds the references ([8b0d687](https://github.com/urmzd/supersource/commit/8b0d68732395f2d882f922b24c5ef1ed5ee06270))
- **ci**: measure child RSS without Python launcher memory ([504dcaa](https://github.com/urmzd/supersource/commit/504dcaa221da84766a860e8e1302a0835c9bbd76))
- **book**: one chapter per topic and single section numbering (#10) ([60f26bc](https://github.com/urmzd/supersource/commit/60f26bceea38395c9894a8cae85cf3793af5e95a))
- **ci**: migrate sr v4 to v7 for artifact and input support (#3) ([5f55293](https://github.com/urmzd/supersource/commit/5f55293a5f8807dc6da61a7da82650514c4d59b9))
- auto-fix lint errors and suppress pre-existing violations in ruff config ([6527edd](https://github.com/urmzd/supersource/commit/6527edda7e7d91df73884b4130f2838ffa191d63))
- add ruff and pytest as dev dependencies ([a80b1d4](https://github.com/urmzd/supersource/commit/a80b1d4514f3e812dddfbed175516cf573540ccd))
- remove --group dev from ci.yml — no dev dependency group defined ([e311c67](https://github.com/urmzd/supersource/commit/e311c6701b69ea86ff50bea5fbf08c70eb76ae70))

### Refactoring

- **llm-systems**: rename the serving platforms guide to match its title ([6d82689](https://github.com/urmzd/supersource/commit/6d826891199828c04e768c6802b6a1d1d1867695))
- **practice**: split build languages into core and optional ([eac259c](https://github.com/urmzd/supersource/commit/eac259cbbad5e1d3306741e81d6f46f8756c8b39))
- **practice**: consolidate predict and build into one practice path ([584c5ff](https://github.com/urmzd/supersource/commit/584c5ffac10c72af5ab965693ca1a6e8eb67295e))
- **algorithms**: move ML practice implementations to canonical location ([a2751db](https://github.com/urmzd/supersource/commit/a2751db7774591bf3e83a668614af77342bf73fe))
- rename project from superpowers to supersource ([63ed7d9](https://github.com/urmzd/supersource/commit/63ed7d95ef9b43a20dfc71dc5513920ce04826e0))
- move algorithms to dedicated directory ([4964e02](https://github.com/urmzd/supersource/commit/4964e021124dbd950db02377582417401ccd59c3))
- move algorithms to dedicated directory ([7e89214](https://github.com/urmzd/supersource/commit/7e89214d86765c1a70e6a5fb7fe848d83d84824f))
- move pattern guides into topic READMEs ([f23191d](https://github.com/urmzd/supersource/commit/f23191dae10c8b2e07a4f21f1f24491589066c3e))
- **ml-stats**: enhance existing ML implementations ([1ea63c8](https://github.com/urmzd/supersource/commit/1ea63c8424e92f5a3f61fae76e5351fbd1119ab8))
- **ml-stats**: update assignment references and framework code ([5092ba8](https://github.com/urmzd/supersource/commit/5092ba808586e295e90260f4c389e1202f1cd0b7))
- **interview**: categorize each section ([45c0a3c](https://github.com/urmzd/supersource/commit/45c0a3c4072480e7f34e1092d30045c66862b670))
- rename example-8.py to example-8.c ([9edbf64](https://github.com/urmzd/supersource/commit/9edbf64aa5362f68cc4244ba2911cd214cf6fad1))

### Misc

- keep uv.lock unchanged in this PR ([a5a70b1](https://github.com/urmzd/supersource/commit/a5a70b13b722ffb6aae0d0be08bef68e5e792261))
- **algorithms**: ruff format files touched by the history scrub ([9181543](https://github.com/urmzd/supersource/commit/918154362b30be199bdc8ae4fbb6cece89e51326))
- **pages**: build the site on PRs and deploy it to supersource.urmzd.com ([cbe7778](https://github.com/urmzd/supersource/commit/cbe7778a8ab46abe6deff84cb8c407ee63a29797))
- **quantization**: add FP format family table to the FP8 section ([93d25b3](https://github.com/urmzd/supersource/commit/93d25b323f573cc31174d6a46c1157e8196692d2))
- **llm-systems**: add serving architecture diagrams in D2 and Mermaid ([f9e1428](https://github.com/urmzd/supersource/commit/f9e14281927bb87c13ca9537d11a2cc19c7187f8))
- **field-engineering**: add the field engineering track ([e973ce2](https://github.com/urmzd/supersource/commit/e973ce21575991ec3ce99a24a4afa73c53bce848))
- **book**: build composed path PDFs and fail on stale diagram renders ([4f231dd](https://github.com/urmzd/supersource/commit/4f231dd2dcbed890384316fa175a5589672c0425))
- **llm-systems**: apply ruff format to the model loading scripts ([d172316](https://github.com/urmzd/supersource/commit/d172316bf4ecb5b8b28aba0e654eb4cd4e461b47))
- add CS curriculum and validated source register ([c3e167c](https://github.com/urmzd/supersource/commit/c3e167ca1cd4579c02e0dedf2d57df4e7eaddc5d))
- **book**: build and release one PDF per role path ([98ad818](https://github.com/urmzd/supersource/commit/98ad81897fbc85bae67ba6ccdba6a8c8245359b2))
- **ml**: add training and post-training topic with LoRA, DPO, and memory tooling ([fe39c0c](https://github.com/urmzd/supersource/commit/fe39c0c4aef1bc9ad6e22337b5c6e5730f0533d0))
- **llm-systems**: add serving and load deep dive with capacity calculator and load generator ([80725f2](https://github.com/urmzd/supersource/commit/80725f22c6b6dc72064e5e085e783d3037272de8))
- **llm-systems**: add model loading deep dive with safetensors and config tooling ([c718eeb](https://github.com/urmzd/supersource/commit/c718eebbb11a23408e690230089506d3fff3c5f0))
- point measure tests at practice/bin after the merge ([ed46981](https://github.com/urmzd/supersource/commit/ed4698191491784a5f810170ce129df320e6623c))
- **llm-systems**: add serving platforms guide and link it from framework docs ([9d44790](https://github.com/urmzd/supersource/commit/9d4479040893e9be4026a9967ebe80e1bf0a8e0f))
- **case-studies**: add Agent Evaluation Harness worked build ([b602a04](https://github.com/urmzd/supersource/commit/b602a045682393e37803d6a75a732f5664478007))
- **platform**: cross-link routing track and serving platforms guide ([3750565](https://github.com/urmzd/supersource/commit/3750565d8dab6790fbea692a533963282786efdc))
- **platform**: cover System One decision models and specialised models in the pool ([b797fda](https://github.com/urmzd/supersource/commit/b797fdaffe87f93183d0c5b23016cd0bfcea2a10))
- **case-studies**: explain what text-to-SQL benchmarks can and cannot tell you ([3936ffa](https://github.com/urmzd/supersource/commit/3936ffacd208147f17279dc198066f818e2150c2))
- **platform**: add Model Routing & Cascades track ([8e91a13](https://github.com/urmzd/supersource/commit/8e91a130e34df9a3de2e221f1316294fd182d613))
- explain LLM serving platforms and routing trade-offs ([b8be21a](https://github.com/urmzd/supersource/commit/b8be21aadf79e17270020824958052c37dd1e698))
- **case-studies**: add worked-build track consolidating four projects ([52f879c](https://github.com/urmzd/supersource/commit/52f879c52a18a00fe7577f789da2fe63580204b9))
- **craftsmanship**: add Lessons from Practice, consolidating the lessons archive ([eb25b5c](https://github.com/urmzd/supersource/commit/eb25b5c2886648971ab9e7bb458a1ddc89272dde))
- bump GitHub Actions to Node 24 majors and switch app token to client-id ([6067345](https://github.com/urmzd/supersource/commit/606734578b5058ecb745bf50af2479612e9ec792))
- bump setup-chrome to v2 for Node 24 support ([625f22e](https://github.com/urmzd/supersource/commit/625f22ed5029e2d5798a7ac356fb6593027fd785))
- pin setup-uv to v7 (no floating v8 major tag exists) ([07e7e6a](https://github.com/urmzd/supersource/commit/07e7e6a8d79467afa17b17c8df5b0ed1b0d53d4e))
- sync uv.lock to 0.5.0 and add it to sr stage_files ([0107e31](https://github.com/urmzd/supersource/commit/0107e31c9f4e11f462d11f623379a73dbe93a220))
- bump all GitHub Actions to latest Node 24 majors ([015698a](https://github.com/urmzd/supersource/commit/015698a3837d45524c876a808e73e332d297f735))
- bump checkout to v5 and setup-uv to v6 for Node 24 support ([fb0de07](https://github.com/urmzd/supersource/commit/fb0de07d12ac6936bc494939b7a88ee4dce341a7))
- **data-engineering**: ruff-format Spark job and sync uv.lock to 0.4.0 ([470cd9a](https://github.com/urmzd/supersource/commit/470cd9acd0c993296a82be45344ea19dd45f5281))
- **infrastructure**: apply ruff format to demo code (#13) ([e330bf2](https://github.com/urmzd/supersource/commit/e330bf2b4e997c7da364edc6008c9cc17c055f26))
- **ml**: add LLM development lifecycle section to Foundation Models (#9) ([5fb7b2a](https://github.com/urmzd/supersource/commit/5fb7b2aaed10f07f0275b62e41354410c0311169))
- attach curriculum PDF as a release asset via sr (#7) ([dea7d33](https://github.com/urmzd/supersource/commit/dea7d33e579227b07bcb605aec96768f0f87a765))
- build the whole curriculum into one learnable PDF (#6) ([111b36f](https://github.com/urmzd/supersource/commit/111b36ff37e432766f3651021571949f233a0002))
- add Diagramming & Operations track (#5) ([d814d8a](https://github.com/urmzd/supersource/commit/d814d8a35fe6b5142f094426201cd0ee39b08629))
- add data engineering track and LLM systems & inference topic (#4) ([b327222](https://github.com/urmzd/supersource/commit/b3272229feeb9d1624feb3d5f1871cfab5ed1bc8))
- **ci**: bump sr to v8 ([d1b724c](https://github.com/urmzd/supersource/commit/d1b724c6460f79d1a504422088296327dd4e406c))
- **ci**: remove unused force input from release workflow ([0699ea0](https://github.com/urmzd/supersource/commit/0699ea00040dffa9d3a7b09541a6ca83918b45e1))
- **fix**: add CI badge to README ([933eff6](https://github.com/urmzd/supersource/commit/933eff6540e67616e79af184166f57de3359ad55))
- update internal references from superpowers to supersource ([97a4414](https://github.com/urmzd/supersource/commit/97a4414723bd580a07879262acdd0a35c65c6fbb))
- **community**: add GitHub community-health files ([d1f9927](https://github.com/urmzd/supersource/commit/d1f99275c90644beb814dacaa20e639c2662d707))
- **fix**: rewrite README with standard structure ([b586982](https://github.com/urmzd/supersource/commit/b5869820c0a14a168486be330651f1f497fddc53))
- migrate sr config and action to v4 ([ea6f2ed](https://github.com/urmzd/supersource/commit/ea6f2ede3a376dc932238c51ba36f33ae97cc5bf))
- add linguist overrides to fix language stats ([32b348c](https://github.com/urmzd/supersource/commit/32b348c79437ccf530bece56c0589b8a26983e36))
- update sr action from v2 to v3 ([e18ae5e](https://github.com/urmzd/supersource/commit/e18ae5e687c000c63cc819b7297d2463ffb26ea7))
- auto-format Python files with ruff ([47ccaab](https://github.com/urmzd/supersource/commit/47ccaab9c686ee5d5d1cb4735898ab1fda77d4a1))
- standardize CI/CD — add sr.yaml, ci.yml, release workflow ([d89c84c](https://github.com/urmzd/supersource/commit/d89c84c41795ec353f47997d4949b64bf7daf6a7))
- **practice**: remove incomplete practice implementations ([8a78ba8](https://github.com/urmzd/supersource/commit/8a78ba8e5f7a2e6d5172d70025310bb2dc996900))
- **software-craftsmanship**: add professional development track documentation ([055b3e9](https://github.com/urmzd/supersource/commit/055b3e9aca4285530396babc3b43f05cfa0becc3))
- **practice**: add polyglot practice guide for nine languages ([8c61aa6](https://github.com/urmzd/supersource/commit/8c61aa6f688b4254c94631bb38462217c33236c4))
- **systems**: add comprehensive README for systems and architecture tracks ([bd746a2](https://github.com/urmzd/supersource/commit/bd746a20dc8d1929a476fe38f6709691642341f5))
- **ml**: add comprehensive README for all machine learning tracks ([5e8b53b](https://github.com/urmzd/supersource/commit/5e8b53bba0000b04a30a6561151f10adad6e7b13))
- **math**: add comprehensive README for all mathematics tracks ([abb2cf0](https://github.com/urmzd/supersource/commit/abb2cf06ddad3525b3a0425b60d7e76320897b56))
- **interviews**: add OpenAI preparation resources ([bf2572d](https://github.com/urmzd/supersource/commit/bf2572deefe9edfa007a9563cd2c895f7192dbee))
- **curriculum**: expand with ML/AI, systems, competitive programming, and information theory tracks ([8a521a4](https://github.com/urmzd/supersource/commit/8a521a4cb820ea48ebbc76f0b08126922f15bc1d))
- add structured 21-day study plan ([ee37269](https://github.com/urmzd/supersource/commit/ee37269fafb608edb0b123cd7820915314e7f022))
- restructure main readme for new curriculum ([2496f54](https://github.com/urmzd/supersource/commit/2496f54985fb1afd35c451fe95d542a640f75d13))
- add agent skill following agentskills.io spec ([b28391c](https://github.com/urmzd/supersource/commit/b28391c4d5e90e38435fb440e36eb04bbef56cd4))
- rewrite root README with linked topics table and 21-day study plan ([8e03393](https://github.com/urmzd/supersource/commit/8e033933b5948759a805436c381e26d1d0309dd7))
- add sensitive paths to .gitignore ([8e59b97](https://github.com/urmzd/supersource/commit/8e59b97da130e0c32e79581dac6fc6cb58656eb7))
- license under Apache 2.0 ([09dbee5](https://github.com/urmzd/supersource/commit/09dbee533551556ea434e0ef4c98491be3763893))
- **ml-stats**: add clustering analysis conclusions ([de7011b](https://github.com/urmzd/supersource/commit/de7011b8758a5e91433f1fb8afcd9dcc7ac38f5e))
- add python version specification and project configuration ([b1abbb9](https://github.com/urmzd/supersource/commit/b1abbb954a6534e6d4fb324ea42c0619478f7d5e))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.9.0...v0.10.0)


## 0.8.0 (2026-10-07)

### Features

- **paths**: split the FDE path into six parts composed by Superstar FDE ([3e4b0ea](https://github.com/urmzd/supersource/commit/3e4b0eaf0d00e31816f291ffc15ebb3a268f5c08))
- **practice**: let paths include other paths in ss learn ([57963b1](https://github.com/urmzd/supersource/commit/57963b1d9bca9960313ceb0e83f22a59e36220c9))

### Misc

- **llm-systems**: add serving architecture diagrams in D2 and Mermaid ([2f2a00a](https://github.com/urmzd/supersource/commit/2f2a00a2698b277db3510f830845009c7eada84b))
- **field-engineering**: add the field engineering track ([3261d29](https://github.com/urmzd/supersource/commit/3261d2989e02835ebd8efa95a1877d367ca7951d))
- **book**: build composed path PDFs and fail on stale diagram renders ([9767d83](https://github.com/urmzd/supersource/commit/9767d83e87985ba2cd48761919906ee623e580c5))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.7.0...v0.8.0)


## 0.7.0 (2026-10-07)

### Features

- **paths**: add the FDE inference role path and wire it into the curriculum ([1056748](https://github.com/urmzd/supersource/commit/10567487ad377d22b18a93500eda595ff3860104))
- **practice**: add ss learn to read role paths stage by stage ([f3ca3a0](https://github.com/urmzd/supersource/commit/f3ca3a06baf003bf987296304fe63779c73061ab))
- **practice**: pin the JDK with mise and resolve it inside ss ([be14be3](https://github.com/urmzd/supersource/commit/be14be3358343d9c7835dccd97ad3e0e03e3548f))
- **practice**: complete the C++ track with five more exercises ([4496480](https://github.com/urmzd/supersource/commit/449648076236fba85d04fbf93654fae0a92f3423))
- **practice**: add five C++ build exercises ([723f09d](https://github.com/urmzd/supersource/commit/723f09d7b2c3f57143d48330687b4bd969815e34))
- **practice**: add ten verified C build exercises ([e8467ed](https://github.com/urmzd/supersource/commit/e8467edcb1d176619a06e04c926bd62ec8d9fb7e))

### Bug Fixes

- **practice**: include stdint.h and memory explicitly so gcc builds the references ([8469e86](https://github.com/urmzd/supersource/commit/8469e8659bd1eee0ea2ca4dfe799e96b3d3f3d70))

### Refactoring

- **llm-systems**: rename the serving platforms guide to match its title ([dfcd60d](https://github.com/urmzd/supersource/commit/dfcd60d565018d67941bb69b11cedb04294ca9be))
- **practice**: split build languages into core and optional ([c70086c](https://github.com/urmzd/supersource/commit/c70086c08c05b2a18ac73c1e8c7cef29e1bf745b))
- **practice**: consolidate predict and build into one practice path ([f96c416](https://github.com/urmzd/supersource/commit/f96c416341c5f930db145e6bdfbf48763d5c76b5))

### Misc

- **llm-systems**: apply ruff format to the model loading scripts ([3b1f5b0](https://github.com/urmzd/supersource/commit/3b1f5b005097d49fda2ecb40046f72291ca63a41))
- add CS curriculum and validated source register ([bc9ac9f](https://github.com/urmzd/supersource/commit/bc9ac9fce7610ecff439659d587dd4e141335745))
- **book**: build and release one PDF per role path ([1844ffb](https://github.com/urmzd/supersource/commit/1844ffb20ea2cc75b9c79147d96889fd874197aa))
- **ml**: add training and post-training topic with LoRA, DPO, and memory tooling ([59cf089](https://github.com/urmzd/supersource/commit/59cf0897cc00e01b663b083ae0908f90ffeffe7b))
- **llm-systems**: add serving and load deep dive with capacity calculator and load generator ([a8b001d](https://github.com/urmzd/supersource/commit/a8b001d4a57b63654fe8232de596312bf9adede3))
- **llm-systems**: add model loading deep dive with safetensors and config tooling ([e65a686](https://github.com/urmzd/supersource/commit/e65a686203f6ba070bce35c2a85d9a0d289b660e))
- point measure tests at practice/bin after the merge ([516ec07](https://github.com/urmzd/supersource/commit/516ec070c71429cd2bac70040ea38b38888e1f74))
- **llm-systems**: add serving platforms guide and link it from framework docs ([f801646](https://github.com/urmzd/supersource/commit/f8016467ce1a031617e63443fd07d6375f0e880a))
- **case-studies**: add Agent Evaluation Harness worked build ([a5c2e2a](https://github.com/urmzd/supersource/commit/a5c2e2a91893461361b99d388fe0742715e92f88))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.6.0...v0.7.0)


## 0.6.0 (2026-09-27)

### Features

- **practice**: assert wall time and peak memory per snippet ([ffb1b74](https://github.com/urmzd/supersource/commit/ffb1b74154ec199b05048edd77df17ef0d59933f))
- **practice**: fill the predict harness with 24 verified snippets ([c08fea3](https://github.com/urmzd/supersource/commit/c08fea3522c5ec950703c304e13308fd0ccacbf9))
- **practice**: add predict-then-run harness baseline ([ae50471](https://github.com/urmzd/supersource/commit/ae50471ea036db911c96443befb3e1df418f3c1e))
- **programming-languages**: add Type Systems & Polymorphism track (#14) ([438d929](https://github.com/urmzd/supersource/commit/438d9295c7dad5940a210105c98a1cbfe1f6b7be))

### Bug Fixes

- **ci**: measure child RSS without Python launcher memory ([b1f3a00](https://github.com/urmzd/supersource/commit/b1f3a00ea4d0583e03af12cde526a2dbb1e0e9dc))

### Misc

- **platform**: cross-link routing track and serving platforms guide ([fc00373](https://github.com/urmzd/supersource/commit/fc003732f91740041b643f6018b0eaf38ee5cb5c))
- **platform**: cover System One decision models and specialised models in the pool ([bd330ca](https://github.com/urmzd/supersource/commit/bd330ca8778a7119dca955f028950260df855954))
- **case-studies**: explain what text-to-SQL benchmarks can and cannot tell you ([5ab5473](https://github.com/urmzd/supersource/commit/5ab54731e6694c48e9ea930cede7bf62257947cf))
- **platform**: add Model Routing & Cascades track ([fda5cc7](https://github.com/urmzd/supersource/commit/fda5cc79e47292fd2df954816388100571755b94))
- explain LLM serving platforms and routing trade-offs ([9fdcacb](https://github.com/urmzd/supersource/commit/9fdcacb6c251bb6414628de4a7c418d62e2862b9))
- **case-studies**: add worked-build track consolidating four projects ([781847d](https://github.com/urmzd/supersource/commit/781847d084809ef186e4040ad26a9c7c65c08da6))
- **craftsmanship**: add Lessons from Practice, consolidating the lessons archive ([426640d](https://github.com/urmzd/supersource/commit/426640d1d95bfcfa3b594feaa328cf9e17ce2c75))
- bump GitHub Actions to Node 24 majors and switch app token to client-id ([0af765e](https://github.com/urmzd/supersource/commit/0af765e43b730c9f86012fec10f55a8e9618afcd))
- bump setup-chrome to v2 for Node 24 support ([12673d4](https://github.com/urmzd/supersource/commit/12673d493b4aafbedad041769b91901aa2e49352))
- pin setup-uv to v7 (no floating v8 major tag exists) ([b6c4188](https://github.com/urmzd/supersource/commit/b6c4188f0efddafb870ead127fff56a53fc4a3e6))
- sync uv.lock to 0.5.0 and add it to sr stage_files ([bd61a10](https://github.com/urmzd/supersource/commit/bd61a1073526c0948c93f9af4b58c696b5d3e207))
- bump all GitHub Actions to latest Node 24 majors ([0a92da2](https://github.com/urmzd/supersource/commit/0a92da2d01c2ca327e6acda915d54616a4fd5d7e))
- bump checkout to v5 and setup-uv to v6 for Node 24 support ([b8ca677](https://github.com/urmzd/supersource/commit/b8ca6774728aa66717087f980c2ae0a625f6980f))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.5.0...v0.6.0)


## 0.5.0 (2026-06-14)

### Features

- **data-engineering**: add runnable Streamflow orders pipeline (Spark + dbt + Airflow) ([4e565e9](https://github.com/urmzd/supersource/commit/4e565e99c86b733c83c5cdff1111534c6b209e7e))

### Misc

- **data-engineering**: ruff-format Spark job and sync uv.lock to 0.4.0 ([b54e9cb](https://github.com/urmzd/supersource/commit/b54e9cb97f51baa746a7db7e2dce9cb72bb6b46a))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.4.0...v0.5.0)


## 0.4.0 (2026-06-14)

### Features

- **infrastructure**: split diagramming-and-operations into focused tracks ([abc2225](https://github.com/urmzd/supersource/commit/abc222582df212a63a58dfaa11cc9cdcf8226e55))

### Misc

- **infrastructure**: apply ruff format to demo code (#13) ([cefb6de](https://github.com/urmzd/supersource/commit/cefb6dedff0d6c1ad827408e74e112ab42e825e3))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.3.1...v0.4.0)


## 0.3.1 (2026-06-14)

### Bug Fixes

- **book**: one chapter per topic and single section numbering (#10) ([a6ed4a1](https://github.com/urmzd/supersource/commit/a6ed4a139fab15cadbc29724e571b596da1e9186))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.3.0...v0.3.1)


## 0.3.0 (2026-06-13)

### Features

- **ml**: Neural Architectures & the History of Deep Learning (#8) ([e3627d1](https://github.com/urmzd/supersource/commit/e3627d19e9240b2168f4f1de0881cae6fafbe935))

### Misc

- **ml**: add LLM development lifecycle section to Foundation Models (#9) ([d60bc49](https://github.com/urmzd/supersource/commit/d60bc49b463fdd0d16e8b9364f57bc9f0442e858))
- attach curriculum PDF as a release asset via sr (#7) ([5ee5cc8](https://github.com/urmzd/supersource/commit/5ee5cc85cda4e80948045f8e2033ed7055139582))
- build the whole curriculum into one learnable PDF (#6) ([ca2a1a6](https://github.com/urmzd/supersource/commit/ca2a1a62600211e318abc3d6b11b3bacd3aade4a))
- add Diagramming & Operations track (#5) ([41074eb](https://github.com/urmzd/supersource/commit/41074eb134426d4102361a2e0c53709fd8bd9df4))
- add data engineering track and LLM systems & inference topic (#4) ([d769f70](https://github.com/urmzd/supersource/commit/d769f70eaeb04a1930351851a840ff1c61d9ac06))
- **ci**: bump sr to v8 ([b3ba9d3](https://github.com/urmzd/supersource/commit/b3ba9d383e8ea9f0586cf0796335c9e553ac1fa0))
- **ci**: remove unused force input from release workflow ([90c4dc5](https://github.com/urmzd/supersource/commit/90c4dc57f51e76193bd7048e04bf6a2d22338d48))
- **fix**: add CI badge to README ([4841e57](https://github.com/urmzd/supersource/commit/4841e57f8fa77ead1b6e6172b4cc882634f99000))
- update internal references from superpowers to supersource ([559cba3](https://github.com/urmzd/supersource/commit/559cba3f901c603cc2f9670f8881eb91d1b5165b))
- **community**: add GitHub community-health files ([0892131](https://github.com/urmzd/supersource/commit/08921314378366be496771e9f101ecee9c2670c5))
- **fix**: rewrite README with standard structure ([ee66c07](https://github.com/urmzd/supersource/commit/ee66c07a757ea1d2fca44120a38af40bd03c8257))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.2.1...v0.3.0)


## 0.2.1 (2026-04-16)

### Bug Fixes

- **ci**: migrate sr v4 to v7 for artifact and input support (#3) ([eefd537](https://github.com/urmzd/supersource/commit/eefd537226c35bd8615ea3a8bf65b1853929a6fd))

### Misc

- migrate sr config and action to v4 ([67a66c3](https://github.com/urmzd/supersource/commit/67a66c3c4b794d35623915ae7fad6dc7cbff98fd))
- add linguist overrides to fix language stats ([8b0132f](https://github.com/urmzd/supersource/commit/8b0132fcce12972090b150d1c9fba18f289a60d6))
- update sr action from v2 to v3 ([ce2dfb6](https://github.com/urmzd/supersource/commit/ce2dfb67d32a391a85f6103dc691effa4acb768e))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.2.0...v0.2.1)


## 0.2.0 (2026-03-30)

### Features

- **swiss-table**: scaffold Swiss Table hash table implementation ([8b15450](https://github.com/urmzd/supersource/commit/8b15450a989777cec383784b073dd3bd9936788e))
- **skills**: add study-session skill for interactive learning ([8645990](https://github.com/urmzd/supersource/commit/86459906d20eeb42a61c7742428392966398e2c0))
- **skills**: add study-plan skill for personalized learning ([2c7a645](https://github.com/urmzd/supersource/commit/2c7a64573030e8d33f6e87aaf33f95626aaa30b8))
- **skills**: add practice-impl skill for scaffolding implementations ([a715ff2](https://github.com/urmzd/supersource/commit/a715ff2081526f6e348f1f490ea6e09739d9f07d))
- add mathematics curriculum foundation ([9f2a1e7](https://github.com/urmzd/supersource/commit/9f2a1e76b40a8f2acc0fd6b04ebc62994f6d6a97))
- **patterns**: add existing pattern guides for all topics ([197b976](https://github.com/urmzd/supersource/commit/197b976e2633ce611b7b389de69b9b04102ce5ba))
- **patterns**: add ML & Statistics patterns guide ([1e56195](https://github.com/urmzd/supersource/commit/1e5619559e4c6c043b9412630e7d2f65b42e680c))
- **interviews**: add comprehensive interview guides for 19 top companies ([84884f9](https://github.com/urmzd/supersource/commit/84884f9e7a9054df3cf36ac288c36d18f1f7f4e8))
- **probabilistic-structures**: add bloom filter event validator ([abb499f](https://github.com/urmzd/supersource/commit/abb499fe190e7d69e2afd9099411ed0df00923aa))
- **ml-stats**: add linear regression implementation ([9de1eda](https://github.com/urmzd/supersource/commit/9de1eda76eb3a88a2c49b365e681be258a8685cb))
- **greedy**: implement Huffman coding compression algorithm ([d1d6f94](https://github.com/urmzd/supersource/commit/d1d6f9462866ea847fb050d4592887eac92ecf15))
- **graphs**: add Charles and the Corgi Conundrum solution template ([fdeb949](https://github.com/urmzd/supersource/commit/fdeb9498efef11049fce520662fbf0551316d317))
- **practice**: add TF-IDF vector search implementation ([e4a5ddd](https://github.com/urmzd/supersource/commit/e4a5ddd7cbe0a2b9fcd81999f6568c8dcf6c602a))
- **practice**: add k-means clustering implementation ([03efa32](https://github.com/urmzd/supersource/commit/03efa32381ebb10ea26161d1e4320a238c648f60))
- **main**: add entry point script ([8abb5b3](https://github.com/urmzd/supersource/commit/8abb5b3430cfca0869ab7db8796d4756df636d19))

### Bug Fixes

- auto-fix lint errors and suppress pre-existing violations in ruff config ([44b7fb1](https://github.com/urmzd/supersource/commit/44b7fb128116e71ec00ee51aefd7aa2e97b8162b))
- add ruff and pytest as dev dependencies ([dcc9cbc](https://github.com/urmzd/supersource/commit/dcc9cbcbc17d88f055f49a8dcbb50ebd62efe679))
- remove --group dev from ci.yml — no dev dependency group defined ([64ea772](https://github.com/urmzd/supersource/commit/64ea77222f3ffe02a8d79db6f5f88a10ad22b6e2))

### Documentation

- **software-craftsmanship**: add professional development track documentation ([bb7f674](https://github.com/urmzd/supersource/commit/bb7f674eaff0bb6eb8b4087ecf7189896bafdd59))
- **practice**: add polyglot practice guide for nine languages ([1a02774](https://github.com/urmzd/supersource/commit/1a027745f2d552da0acea698539422dd27e59728))
- **systems**: add comprehensive README for systems and architecture tracks ([f4cb6e2](https://github.com/urmzd/supersource/commit/f4cb6e2a5c0a293bea6dc896dacc426b7ae39e60))
- **ml**: add comprehensive README for all machine learning tracks ([51e3fdf](https://github.com/urmzd/supersource/commit/51e3fdf4088a65d2bfd743264b9a222b25a6af6f))
- **math**: add comprehensive README for all mathematics tracks ([b722838](https://github.com/urmzd/supersource/commit/b7228386dc33f6ad0d2674cb72b6ec228d92db7f))
- **interviews**: add OpenAI preparation resources ([2da8c50](https://github.com/urmzd/supersource/commit/2da8c501ea641134fa206ea17e52be54a5238811))
- **curriculum**: expand with ML/AI, systems, competitive programming, and information theory tracks ([c5588c5](https://github.com/urmzd/supersource/commit/c5588c5115d2b6d20ab3b2253956d8bb284e9d32))
- add structured 21-day study plan ([c513458](https://github.com/urmzd/supersource/commit/c513458504aeee62c7cbaa7934cfc9fefef29497))
- restructure main readme for new curriculum ([115d5cb](https://github.com/urmzd/supersource/commit/115d5cb042cbdda309373e586c03e085c3ce5466))
- add agent skill following agentskills.io spec ([3ac34ca](https://github.com/urmzd/supersource/commit/3ac34ca46aefd294e5536741501fb9e97f3c6033))
- rewrite root README with linked topics table and 21-day study plan ([de08ad2](https://github.com/urmzd/supersource/commit/de08ad2db1f2c9da05195d2f560e35fe345529b8))
- **ml-stats**: add clustering analysis conclusions ([9ae32c7](https://github.com/urmzd/supersource/commit/9ae32c747d8233fc38686b8268e7dc3ac5f7d71b))

### Refactoring

- **algorithms**: move ML practice implementations to canonical location ([0d2232d](https://github.com/urmzd/supersource/commit/0d2232d012f26011b6903985147d3e00884f3992))
- rename project from superpowers to supersource ([2e7380b](https://github.com/urmzd/supersource/commit/2e7380b3b572b657bd420dc51cae31816a08e534))
- move algorithms to dedicated directory ([fafdc4e](https://github.com/urmzd/supersource/commit/fafdc4ebe67b43e2b145397c587ecc0aaa4fc0b4))
- move algorithms to dedicated directory ([462ee88](https://github.com/urmzd/supersource/commit/462ee88c6febc884162ff0e20503fa87417a57fb))
- move pattern guides into topic READMEs ([3bbdcaf](https://github.com/urmzd/supersource/commit/3bbdcaf1cbc015910854372a1c5210848cbd52e9))
- **ml-stats**: enhance existing ML implementations ([237e02e](https://github.com/urmzd/supersource/commit/237e02e79b5f88e27ba3e4d6bd27c0f75a5a5a87))
- **ml-stats**: update assignment references and framework code ([ab9cc18](https://github.com/urmzd/supersource/commit/ab9cc1835377586ae034f4dfc582bf111218de0c))
- **interview**: categorize each section ([121fca5](https://github.com/urmzd/supersource/commit/121fca57734b6438b20b8e189c1c28ef55c79589))
- rename example-8.py to example-8.c ([2164e3d](https://github.com/urmzd/supersource/commit/2164e3df56e00ff9256ec06447f6133b7b2359b2))

### Miscellaneous

- auto-format Python files with ruff ([05e7c9a](https://github.com/urmzd/supersource/commit/05e7c9a9a3be0c866f79c4aff68fa5bf34085723))
- standardize CI/CD — add sr.yaml, ci.yml, release workflow ([d8b2505](https://github.com/urmzd/supersource/commit/d8b25050910588b3354028294a3645b8cc35d863))
- **practice**: remove incomplete practice implementations ([4a354da](https://github.com/urmzd/supersource/commit/4a354daffe5b0130d21b9e97e1ccc7598dbd6f04))
- add sensitive paths to .gitignore ([73358a5](https://github.com/urmzd/supersource/commit/73358a58683a0b352a061d879bb33b9251942986))
- license under Apache 2.0 ([b4168d6](https://github.com/urmzd/supersource/commit/b4168d64b3b13113b74bc4435239c4243d9b18fc))
- add python version specification and project configuration ([510f303](https://github.com/urmzd/supersource/commit/510f303f6fec29441cb665484f70860dd817d406))

[Full Changelog](https://github.com/urmzd/supersource/compare/v0.1.0...v0.2.0)
