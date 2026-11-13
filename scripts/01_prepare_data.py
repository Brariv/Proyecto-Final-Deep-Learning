"""Paso 1: lee los CSV de IBM y genera los arreglos procesados (variables sin fuga de información)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fraudscan import prepare
from fraudscan.config import base_parser, load_config

if __name__ == "__main__":
    parser = base_parser(__doc__)
    parser.add_argument("--nrows", type=int, default=None, help="Leer solo las primeras N filas (pruebas)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.profile)
    prepare.build(cfg, nrows=args.nrows)
