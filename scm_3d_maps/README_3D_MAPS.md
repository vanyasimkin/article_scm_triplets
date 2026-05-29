# SCM 3D orientation maps

Этот модуль добавляет новый этап расчётов к проекту `article_scm_triplets`: построение 3D-карт парного и трёхчастичного взаимодействия диэлектрических сфер во вращающемся электрическом поле методом матричного SCM.

Модуль предполагает, что в корне репозитория уже есть валидированный файл:

```text
scm_core.py
```

и что запуск выполняется из корня проекта:

```text
article_scm_triplets/
```

---

## 1. Что считаем физически

Внешнее поле вращается в плоскости `xy`:

```text
E_k = E0 (cos theta_k, sin theta_k, 0),
theta_k = 2*pi*k/N_theta.
```

Для пары считается ориентационно-зависимая парная избыточная энергия:

```text
phi_pair(r, beta) = < U2(r,beta,k) - 2 U1(k) >_k.
```

Здесь:

- `r` — расстояние между центрами частиц;
- `beta` — угол между межцентровым вектором пары и плоскостью вращения поля `xy`;
- `beta = 0 deg` — пара лежит в плоскости поля;
- `beta = 90 deg` — пара ориентирована вдоль оси `z`.

Для тройки считается:

```text
Phi3 = < U3(k) - 3 U1(k) >_k.
```

Парно-аддитивная оценка:

```text
Phi3_pairwise = sum_{i<j} phi_pair(r_ij, beta_ij).
```

Непарный трёхчастичный вклад:

```text
Delta3 = Phi3 - Phi3_pairwise.
```

Относительная неаддитивность:

```text
eta3_pair = |Delta3| / |Phi3_pairwise|.
```

Симметричная нормировка:

```text
eta3_sym = 2|Delta3| / (|Phi3| + |Phi3_pairwise|).
```

Главный физический объект статьи:

```text
Delta3(r12, r13, gamma, psi, alpha).
```

Он показывает, где многочастичная переполяризация усиливает притяжение относительно суммы пар (`Delta3 < 0`), а где ослабляет его (`Delta3 > 0`).

---

## 2. Геометрия пары

Пара задаётся центрами:

```text
r1 = -(r/2) n_beta,
r2 = +(r/2) n_beta,
```

где

```text
n_beta = (cos beta, 0, sin beta).
```

---

## 3. Геометрия тройки

Частица 1 является центральной:

```text
r1 = (0, 0, 0).
```

В исходной локальной плоскости тройки:

```text
r2 = r12 (cos alpha, sin alpha, 0),
r3 = r13 (cos(alpha + gamma), sin(alpha + gamma), 0).
```

Затем вся плоскость тройки наклоняется относительно плоскости вращения поля на угол `psi` поворотом вокруг оси `x`:

```text
r_i(psi, alpha) = R_x(psi) r_i^(0),
```

где

```text
R_x(psi) = [[1, 0, 0],
            [0, cos psi, -sin psi],
            [0, sin psi,  cos psi]].
```

Параметры тройки:

```text
r12, r13, gamma, psi, alpha.
```

Расстояние между частицами 2 и 3:

```text
r23 = sqrt(r12^2 + r13^2 - 2 r12 r13 cos gamma).
```

Угол каждого ребра относительно плоскости поля:

```text
beta_ij = asin(|e_ij · e_z|),
e_ij = (r_j - r_i)/|r_j - r_i|.
```

---

## 4. Структура файлов

После распаковки в корень репозитория структура должна быть такой:

```text
article_scm_triplets/
│
├── scm_core.py
├── scm_config.py
├── ...
│
└── scm_3d_maps/
    ├── README_3D_MAPS.md
    ├── __init__.py
    ├── scm3d_config.py
    ├── scm3d_geometry.py
    ├── scm3d_utils.py
    ├── scm3d_pair_map.py
    ├── scm3d_triangle_map.py
    ├── scm3d_convergence.py
    ├── scm3d_analysis.py
    │
    ├── scripts/
    │   ├── __init__.py
    │   ├── run_00_pair_orientation_map.py
    │   ├── run_01_triangle_isosceles_map.py
    │   ├── run_02_triangle_asymmetric_map.py
    │   ├── run_03_lmax_convergence_checks.py
    │   └── analyze_3d_maps.py
    │
    └── notebooks/
        ├── 00_pair_orientation_map.ipynb
        ├── 01_triangle_isosceles_map.ipynb
        ├── 02_triangle_asymmetric_map.ipynb
        ├── 03_lmax_convergence_checks.ipynb
        └── 04_analysis_figures.ipynb
```

Результаты сохраняются не внутри `scm_3d_maps/`, а в корне проекта:

```text
article_scm_triplets/results_scm_3d_maps/
```

---

## 5. Расчётные стадии

### Stage 1 — pair orientation map

Скрипт:

```bash
python -m scm_3d_maps.scripts.run_00_pair_orientation_map
```

Сетка:

```text
r/d = 1.10, 1.20, 1.35, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 8.00, 10.00
beta = 0, 15, 30, 45, 60, 75, 90 deg
```

Сохраняет:

```text
results_scm_3d_maps/scm_pair_orientation_map_lmax6.npz
```

#### Содержимое таблицы `.npz`

```text
r_over_d              (n_r,)
r                     (n_r,)             физическое расстояние, м
beta_deg              (n_beta,)
lmax                  scalar
U_single_k            (n_orient,)         энергия одиночной сферы для фаз поля
U_single_analytic_k   (n_orient,)         аналитическая проверка одиночной сферы
U_pair_rbk            (n_r, n_beta, n_orient)
phi_pair_rbk          (n_r, n_beta, n_orient)
phi_pair_avg_rb       (n_r, n_beta)
n_orient, n_quad, a, d, E0, eps0, eps1_r, eps2_r, timestamp
```

Главная величина:

```text
phi_pair_avg_rb[ir, ib] = < U2(r,beta,k) - 2U1(k) >_k.
```

---

### Stage 2 — main isosceles triangle map

Скрипт:

```bash
python -m scm_3d_maps.scripts.run_01_triangle_isosceles_map
```

Здесь:

```text
r12 = r13 = r.
```

Сетка:

```text
r/d    = 1.10, 1.20, 1.35, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00
gamma  = 60, 75, 90, 105, 120, 135, 150, 165, 180 deg
psi    = 0, 15, 30, 45, 60, 75, 90 deg
alpha  = 0, 15, 30, 45, 60, 75, 90 deg
```

Всего:

```text
10 * 9 * 7 * 7 = 4410 geometry points.
```

Сохраняет:

```text
results_scm_3d_maps/scm_triangle_isosceles_map_lmax6.npz
```

#### Содержимое таблицы `.npz`

```text
r_over_d              (n_r,)
r                     (n_r,)
gamma_deg             (n_gamma,)
psi_deg               (n_psi,)
alpha_deg             (n_alpha,)
lmax                  scalar
U_single_k            (n_orient,)
U_single_analytic_k   (n_orient,)
U_triplet_rgpak       (n_r, n_gamma, n_psi, n_alpha, n_orient)
U_parts_rgpakp        (n_r, n_gamma, n_psi, n_alpha, n_orient, 3)
Phi3_rgpak            (n_r, n_gamma, n_psi, n_alpha, n_orient)
Phi3_avg_rgpa         (n_r, n_gamma, n_psi, n_alpha)
Phi_pairwise_rgpa     (n_r, n_gamma, n_psi, n_alpha)
Delta3_rgpa           (n_r, n_gamma, n_psi, n_alpha)
eta3_pair_rgpa        (n_r, n_gamma, n_psi, n_alpha)
eta3_sym_rgpa         (n_r, n_gamma, n_psi, n_alpha)
edge_dist_rgpae       (n_r, n_gamma, n_psi, n_alpha, 3)
edge_beta_rgpae       (n_r, n_gamma, n_psi, n_alpha, 3)
min_gap_rgpa          (n_r, n_gamma, n_psi, n_alpha)
```

Индексы в имени массива:

```text
r     -> r/d index
g     -> gamma index
p     -> psi index
a     -> alpha index
k     -> field-orientation index
e     -> edge index: 12, 13, 23
p at the end of U_parts -> particle index: 1, 2, 3
```

---

### Stage 3 — full general triangle map

Скрипт:

```bash
python -m scm_3d_maps.scripts.run_02_triangle_asymmetric_map
```

Исторически файл называется `run_02_triangle_asymmetric_map.py`, но текущая версия выполняет **полный скан общей треугольной тройки** по двум независимым расстояниям:

```text
r12 = расстояние от частицы 1 до частицы 2,
r13 = расстояние от частицы 1 до частицы 3.
```

Сетка:

```text
r12/d = 1.10, 1.20, 1.35, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00
r13/d = 1.10, 1.20, 1.35, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00
gamma = 60, 75, 90, 105, 120, 135, 150, 165, 180 deg
psi   = 0, 15, 30, 45, 60, 75, 90 deg
alpha = 0, 15, 30, 45, 60, 75, 90 deg
```

Всего:

```text
10 * 10 * 9 * 7 * 7 = 44100 geometry points.
```

С учётом `N_theta = 8` это соответствует `352800` SCM-решениям энергии. Это большой расчёт, поэтому скрипт сохраняет промежуточный `.npz` после завершения каждого блока `r12`.

Сохраняет:

```text
results_scm_3d_maps/scm_triangle_full_map_lmax6.npz
```

#### Содержимое таблицы `.npz`

```text
r12_over_d            (n_r12,)
r13_over_d            (n_r13,)
r12                   (n_r12,)
r13                   (n_r13,)
gamma_deg             (n_gamma,)
psi_deg               (n_psi,)
alpha_deg             (n_alpha,)
lmax                  scalar
U_single_k            (n_orient,)
U_single_analytic_k   (n_orient,)
U_triplet_rrgpak      (n_r12, n_r13, n_gamma, n_psi, n_alpha, n_orient)
Phi3_avg_rrgpa        (n_r12, n_r13, n_gamma, n_psi, n_alpha)
Phi_pairwise_rrgpa    (n_r12, n_r13, n_gamma, n_psi, n_alpha)
Delta3_rrgpa          (n_r12, n_r13, n_gamma, n_psi, n_alpha)
eta3_pair_rrgpa       (n_r12, n_r13, n_gamma, n_psi, n_alpha)
eta3_sym_rrgpa        (n_r12, n_r13, n_gamma, n_psi, n_alpha)
edge_dist_rrgpae      (n_r12, n_r13, n_gamma, n_psi, n_alpha, 3)
edge_beta_rrgpae      (n_r12, n_r13, n_gamma, n_psi, n_alpha, 3)
min_gap_rrgpa         (n_r12, n_r13, n_gamma, n_psi, n_alpha)
```

Индексы:

```text
r    -> r12 index
r    -> r13 index
g    -> gamma index
p    -> psi index
a    -> alpha index
k    -> field-orientation index
e    -> edge index: 12, 13, 23
```

Физически частицы 2 и 3 одинаковы, поэтому обмен `r12 <-> r13` задаёт эквивалентную форму треугольника с перенумерацией внешних частиц. Полный квадрат `r12 × r13` сохранён намеренно: он позволяет затем проверить численную симметрию и не пересчитывать дополнительные асимметричные случаи.

---

### Stage 4 — selected lmax convergence checks

Скрипт:

```bash
python -m scm_3d_maps.scripts.run_03_lmax_convergence_checks
```

Считает выбранные пары и тройки при:

```text
lmax = 1, 2, 3, 4, 5, 6.
```

Сохраняет:

```text
results_scm_3d_maps/scm_lmax_convergence_checks.npz
```

#### Содержимое таблицы `.npz`

```text
lmax_list             (n_lmax,)
pair_cases            (n_pair_cases, 2)      columns: r_over_d, beta_deg
triangle_cases        (n_triangle_cases, 5)  columns: r12_over_d, r13_over_d, gamma_deg, psi_deg, alpha_deg
U_single_lk           (n_lmax, n_orient)
pair_phi_lck          (n_lmax, n_pair_cases, n_orient)
pair_phi_lc           (n_lmax, n_pair_cases)
tri_Phi3_lc           (n_lmax, n_triangle_cases)
tri_pairwise_lc       (n_lmax, n_triangle_cases)
tri_Delta3_lc         (n_lmax, n_triangle_cases)
tri_eta_pair_lc       (n_lmax, n_triangle_cases)
tri_eta_sym_lc        (n_lmax, n_triangle_cases)
```

---

## 6. Анализ результатов

После расчётов запустить:

```bash
python -m scm_3d_maps.scripts.analyze_3d_maps
```

Создаёт CSV-таблицы:

```text
results_scm_3d_maps/tables/pair_orientation_summary.csv
results_scm_3d_maps/tables/triangle_isosceles_summary.csv
results_scm_3d_maps/tables/triangle_full_summary.csv
results_scm_3d_maps/tables/lmax_convergence_summary.csv
```

Создаёт рисунки:

```text
results_scm_3d_maps/figures/fig_pair_phi_map_r_beta.png
results_scm_3d_maps/figures/fig_pair_far_tail_r3_phi.png
results_scm_3d_maps/figures/fig_isosceles_Delta3_gamma_psi_r1.20_alpha0.png
results_scm_3d_maps/figures/fig_isosceles_eta3_gamma_psi_r1.20_alpha0.png
results_scm_3d_maps/figures/fig_isosceles_Delta3_vs_r_alpha0_psi0.png
results_scm_3d_maps/figures/fig_lmax_convergence_triangle_Delta3.png
```

Фактические имена карт для разных `alpha` будут содержать значение `alpha` в имени файла.

---

## 7. Рекомендуемый порядок запуска

Сначала обязательно pair map:

```bash
python -m scm_3d_maps.scripts.run_00_pair_orientation_map
```

Потом основная треугольная карта:

```bash
python -m scm_3d_maps.scripts.run_01_triangle_isosceles_map
```

Затем полный скан общей треугольной карты:

```bash
python -m scm_3d_maps.scripts.run_02_triangle_asymmetric_map
```

Проверка сходимости по мультиполям:

```bash
python -m scm_3d_maps.scripts.run_03_lmax_convergence_checks
```

Анализ:

```bash
python -m scm_3d_maps.scripts.analyze_3d_maps
```

---

## 8. Git workflow для новой ветки

Из корня репозитория:

```bash
git checkout -b feature/scm-3d-orientation-maps
```

После копирования папки `scm_3d_maps/`:

```bash
git status
git add scm_3d_maps
git commit -m "Add 3D SCM orientation maps"
git push -u origin feature/scm-3d-orientation-maps
```

Результаты расчётов лучше не коммитить. Добавить в `.gitignore`:

```gitignore
results_scm_3d_maps/*.npz
results_scm_3d_maps/*.npy
results_scm_3d_maps/logs/
```

Если хочется сохранять лёгкие картинки и таблицы в GitHub, не игнорировать:

```text
results_scm_3d_maps/figures/
results_scm_3d_maps/tables/
```

---

## 9. Какой результат ожидаем для статьи

Ожидаем получить ориентационно-зависимую карту многочастичной поляризационной поправки:

```text
Delta3 = Delta3(r12, r13, gamma, psi, alpha).
```

Физически важно увидеть:

1. где `Delta3 < 0`, то есть трёхчастичный вклад усиливает притяжение;
2. где `Delta3 > 0`, то есть трёхчастичный вклад ослабляет притяжение;
3. как граница `Delta3 = 0` зависит от формы тройки и её ориентации относительно плоскости вращения поля;
4. насколько велика относительная неаддитивность `eta3_pair`;
5. где дипольного уровня `lmax=1` недостаточно и нужны мультиполи до `lmax=6`.

Главная формулировка для статьи:

> Во вращающемся электрическом поле непарная трёхчастичная поляризационная поправка является функцией не только расстояний между частицами, но также формы треугольного мотива и его пространственной ориентации относительно плоскости вращения поля.
