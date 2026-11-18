# Release history / История версий

## English

The public repository is [biscet/DraftLab](https://github.com/biscet/DraftLab).
Version 1.0.0 starts this source repository's history independently of model
weights and dataset versions.

- `main` contains v2.0.0 with D-PACE training and the offline Whittle EAGLE-3 workflow.
- The branch `v1.0.0` preserves the original first release.

Before the next release, update VERSION and CHANGELOG and review the intended
source changes. Run the relevant tests and the source privacy check, then commit
the reviewed source and publish it to `main`.

```bash
git switch main
python3 -m unittest discover -s tests -v
python3 coach.py privacy-check
git status --short
git diff --cached
git commit -m "Release DFlash Coach v2.1.0"
git push origin main
```

Stage the intended files before committing. Keep private datasets, weights,
credentials and run artifacts out of Git. Do not force-push over a published
release snapshot.

## Русский

Публичный репозиторий — [biscet/DraftLab](https://github.com/biscet/DraftLab).
Версия 1.0.0 начинает историю исходников независимо от версий весов и датасетов.

- В `main` опубликована версия v2.0.0 с обучением D-PACE и офлайн-сценарием Whittle EAGLE-3.
- Ветка `v1.0.0` сохраняет исходный первый релиз.

Перед выпуском обновите VERSION и CHANGELOG, проверьте состав изменений,
необходимые тесты и privacy-check. Сохраните проверенные изменения в коммите
и отправьте его в `main`. Нужные файлы следует заранее добавить в staging.

Не добавляйте в Git приватные датасеты, веса, ключи и результаты рабочих run.
Не перезаписывайте опубликованные срезы через force-push.
