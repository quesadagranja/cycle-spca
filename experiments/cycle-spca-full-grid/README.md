# Grid completo de CycleSPCA

Este paquete ejecuta el experimento factorial acordado directamente sobre el
código de `quesadagranja/cycle-spca`, sin copiar ni modificar el optimizador.
La configuración de producción contiene:

- 20 valores de `lambda_l1` y 20 de `lambda_tv`;
- `K = 5, 10, 15, 20, 25, 30`;
- `N = 5 000, 10 000, 20 000, 40 000, 80 000`;
- cinco repeticiones aleatorias reproducibles;
- filtro previo obligatorio `imputed <= 72`;
- muestras anidadas dentro de cada repetición;
- `outer_max_iter = 500` e `inner_max_iter = 50 000`;
- un PNG independiente por componente;
- 90 procesos y un hilo BLAS/OpenMP por proceso.

Son 60 000 ajustes y, si todos mantienen su rango nominal, hasta 1 050 000 PNG.

## Lanzamiento inmediato

Las rutas predeterminadas son las actuales del cluster:

```text
repositorio: /home/cquesada/pca
dataset:     /home/cquesada/pca/dataset/matrix_normalized.npy
resultados:  /scratch/cquesada/cycle_spca_results/full_grid_v1
```

Si alguna ruta difiere, edita únicamente esos tres campos al principio de
`grid.json`. Después:

```bash
cd cycle-spca-full-grid
python -m pip install -r requirements.txt
python run_full_grid.py --config grid.json validate
```

Para dejarlo corriendo en `tmux`:

```bash
tmux new -s cyclespca-grid
cd cycle-spca-full-grid
bash launch_90_workers.sh
```

Se sale de `tmux` con `Ctrl-b`, después `d`. El mismo comando reanuda el
experimento: omite todos los ajustes que ya tengan un resultado completo.

## Consultar resultados mientras se calculan

Desde otra terminal:

```bash
cd cycle-spca-full-grid
python run_full_grid.py --config grid.json status
```

Para actualizarlo cada minuto:

```bash
python run_full_grid.py --config grid.json watch --interval 60
```

El supervisor actualiza sin interrumpir a los workers:

```text
/scratch/cquesada/cycle_spca_results/full_grid_v1/tables/fits.csv
/scratch/cquesada/cycle_spca_results/full_grid_v1/tables/results.sqlite
/scratch/cquesada/cycle_spca_results/full_grid_v1/dashboard.html
```

`fits.csv` se reemplaza atómicamente cada minuto. `results.sqlite` usa modo WAL:
puede abrirse en modo lectura durante el cálculo. El CSV por componente se
comprime como `components.csv.gz` y se actualiza con menor frecuencia porque
puede llegar a superar el millón de filas.

Para buscar ajustes terminados y localizar sus PNG:

```bash
python run_full_grid.py --config grid.json query --N 20000 --K 10 \
  --lambda-l1 1.5 --lambda-tv 2.0
```

Para ver el panel en el navegador:

```bash
python run_full_grid.py --config grid.json serve --port 8765
```

En el ordenador local abre el túnel:

```bash
ssh -L 8765:127.0.0.1:8765 cquesada@SERVIDOR
```

y visita `http://127.0.0.1:8765/dashboard.html`.

## Organización y seguridad frente a procesos simultáneos

Cada combinación tiene una carpeta exclusiva:

```text
fits/repeat_01/N_020000/K_10/l1_11/ltv_12/
├── config.json
├── metrics.json
├── components.csv
├── history.csv.gz
├── loadings.npz
├── component_png/
│   ├── component_01.png
│   └── ...
└── DONE
```

El worker construye primero la carpeta en `tmp/` y la publica mediante un
renombrado atómico solo cuando está completa. Ningún worker escribe en los CSV
centrales ni en la carpeta de otro ajuste. Un monitor único incorpora los
registros terminados a SQLite y regenera los CSV.

El dataset se abre con `mmap`. Cada proceso materializa solo el bloque de filas
que CycleSPCA está usando; no se cargan 90 copias completas. Los índices de las
cinco muestras maestras se conservan en `samples/` y cada tamaño `N` es un
prefijo del siguiente.

## Correspondencia temporal de los mapas

`grid.json` fija `order = "F"`. Las columnas del dataset están en orden
cronológico, por lo que la hora cambia primero, seguida del día y la semana.
`loading_tensors()` devuelve `(hora, día, semana, componente)` y cada PNG usa:

```python
heatmap = tensor.transpose(0, 2, 1).reshape(24, 52 * 7)
```

Así, la columna visual es exactamente `semana * 7 + día`; no se realiza ningún
`reshape` ambiguo.

## Estabilidad

Después de completar los ajustes disponibles, el mismo lanzamiento calcula:

- estabilidad local con los vecinos inmediatos de la rejilla de lambdas;
- estabilidad entre las cinco repeticiones de una misma configuración.

Las componentes se emparejan por asignación húngara y coseno absoluto. La
estabilidad penalizada divide la suma de similitudes por el `K` nominal, de
modo que las componentes inactivas o desaparecidas aportan cero y no inflan la
medida. Esta fase también es paralela, reanudable y consultable.

## Comandos auxiliares

```bash
# Preparar hashes y muestras sin lanzar ajustes
python run_full_grid.py --config grid.json prepare

# Regenerar tablas y gráficos agregados en cualquier momento
python run_full_grid.py --config grid.json aggregate --components --plots

# Reanudar solo la estabilidad
python run_full_grid.py --config grid.json stability --workers 90
```

El experimento registra el hash SHA-256 del dataset, el commit exacto del
repositorio, las semillas, los índices de muestra, versiones de software,
historiales de optimización y configuración completa. Si el commit, el dataset
o la configuración científica cambian, el programa se detiene y exige una
nueva carpeta de resultados.
