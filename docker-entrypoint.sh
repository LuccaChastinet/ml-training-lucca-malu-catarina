#!/bin/sh
set -e

python /app/training.py "$@"

if [ -t 0 ] && [ -t 1 ]; then
    printf '\nTreinamento concluído. Os resultados estão em /app/results.\n'
    exec /bin/sh
fi
