import os

from experiments.utils import LoadFromFile

DEFAULT_SAVE_DIR = os.path.join(os.getcwd(), "3DcoordsAtomsBonds_0")

if not os.path.exists(DEFAULT_SAVE_DIR):
    os.makedirs(DEFAULT_SAVE_DIR)


def add_arguments(parser):
    """Helper function to fill the parser object.

    Args:
        parser: Parser object
    Returns:
        parser: Updated parser object
    """

    # Load yaml file
    parser.add_argument(
        "--conf", "-c", type=open, action=LoadFromFile, help="Configuration yaml file"
    )  # keep first

    # Load from checkpoint
    parser.add_argument("--load-ckpt", default=None, type=str)
    parser.add_argument("--load-ckpt-from-pretrained", default=None, type=str)

    # DATA and FILES
    parser.add_argument("-s", "--save-dir", default=DEFAULT_SAVE_DIR, type=str)
    parser.add_argument("--test-save-dir", default=DEFAULT_SAVE_DIR, type=str)

    parser.add_argument(
        "--dataset",
        default="drugs",
        choices=[
            "qm9",
            "drugs",
            "aqm",
            "aqm_qm7x",
            "pcqm4mv2",
            "pepconf",
            "crossdocked",
        ],
    )
    parser.add_argument(
        "--dataset-root", default="----"
    )
    parser.add_argument("--use-adaptive-loader", default=True, action="store_true")
    parser.add_argument("--remove-hs", default=False, action="store_true")
    parser.add_argument("--select-train-subset", default=False, action="store_true")
    parser.add_argument("--train-size", default=0.8, type=float)
    parser.add_argument("--val-size", default=0.1, type=float)
    parser.add_argument("--test-size", default=0.1, type=float)

    parser.add_argument("--dropout-prob", default=0.3, type=float)

    # LEARNING
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("-ib", "--inference-batch-size", default=32, type=int)
    parser.add_argument("--gamma", default=0.975, type=float)
    parser.add_argument("--grad-clip-val", default=10.0, type=float)
    parser.add_argument(
        "--lr-scheduler",
        default="reduce_on_plateau",
        choices=["reduce_on_plateau", "cosine_annealing", "cyclic"],
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["adam", "sgd"],
    )
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--lr-min", default=5e-5, type=float)
    parser.add_argument("--lr-step-size", default=10000, type=int)
    parser.add_argument("--lr-frequency", default=5, type=int)
    parser.add_argument("--lr-patience", default=20, type=int)
    parser.add_argument("--lr-cooldown", default=5, type=int)
    parser.add_argument("--lr-factor", default=0.75, type=float)

    # MODEL
    parser.add_argument("--sdim", default=256, type=int)
    parser.add_argument("--vdim", default=64, type=int)
    parser.add_argument("--latent_dim", default=None, type=int)
    parser.add_argument("--rbf-dim", default=32, type=int)
    parser.add_argument("--edim", default=32, type=int)
    parser.add_argument("--edge-mp", default=False, action="store_true")
    parser.add_argument("--vector-aggr", default="mean", type=str)
    parser.add_argument("--num-layers", default=7, type=int)
    parser.add_argument("--fully-connected", default=True, action="store_true")
    parser.add_argument("--local-global-model", default=False, action="store_true")
    parser.add_argument("--local-edge-attrs", default=False, action="store_true")
    parser.add_argument("--use-cross-product", default=False, action="store_true")
    parser.add_argument("--cutoff-local", default=7.0, type=float)
    parser.add_argument("--cutoff-global", default=10.0, type=float)
    parser.add_argument("--energy-training", default=False, action="store_true")
    parser.add_argument("--property-training", default=False, action="store_true")
    parser.add_argument(
        "--regression-property",
        default="polarizability",
        type=str,
        choices=[
            "dipole_norm",
            "total_energy",
            "HOMO-LUMO_gap",
            "dispersion",
            "atomisation_energy",
            "polarizability",
        ],
    )
    parser.add_argument("--energy-loss", default="l2", type=str, choices=["l2", "l1"])
    parser.add_argument("--use-pos-norm", default=False, action="store_true")

    # For Discrete: Include more features: (is_aromatic, is_in_ring, hybridization)
    parser.add_argument("--additional-feats", default=False, action="store_true")
    parser.add_argument("--use-qm-props", default=False, action="store_true")
    parser.add_argument("--build-mol-with-addfeats", default=False, action="store_true")

    # DIFFUSION
    parser.add_argument(
        "--continuous",
        default=False,
        action="store_true",
        help="If the diffusion process is applied on continuous time variable. Defaults to False",
    )
    parser.add_argument(
        "--noise-scheduler",
        default="cosine",
        choices=["linear", "cosine", "quad", "sigmoid", "adaptive", "linear-time"],
    )
    parser.add_argument("--eps-min", default=1e-3, type=float)
    parser.add_argument("--beta-min", default=1e-4, type=float)
    parser.add_argument("--beta-max", default=2e-2, type=float)
    parser.add_argument("--timesteps", default=500, type=int)
    parser.add_argument("--max-time", type=str, default=None)
    parser.add_argument("--lc-coords", default=3.0, type=float)
    parser.add_argument("--lc-atoms", default=0.4, type=float)
    parser.add_argument("--lc-bonds", default=2.0, type=float)
    parser.add_argument("--lc-charges", default=1.0, type=float)
    parser.add_argument("--lc-mulliken", default=1.5, type=float)
    parser.add_argument("--lc-wbo", default=2.0, type=float)

    parser.add_argument("--pocket-noise-std", default=0.1, type=float)
    parser.add_argument(
        "--use-ligand-dataset-sizes", default=False, action="store_true"
    )

    parser.add_argument(
        "--loss-weighting",
        default="snr_t",
        choices=["snr_s_t", "snr_t", "exp_t", "expt_t_half", "uniform"],
    )
    parser.add_argument("--snr-clamp-min", default=0.05, type=float)
    parser.add_argument("--snr-clamp-max", default=1.50, type=float)

    parser.add_argument(
        "--ligand-pocket-interaction", default=False, action="store_true"
    )
    parser.add_argument("--diffusion-pretraining", default=False, action="store_true")
    parser.add_argument(
        "--continuous-param", default="data", type=str, choices=["data", "noise"]
    )
    parser.add_argument("--atoms-categorical", default=False, action="store_true")
    parser.add_argument("--bonds-categorical", default=False, action="store_true")

    parser.add_argument("--atom-type-masking", default=False, action="store_true")
    parser.add_argument("--use-absorbing-state", default=False, action="store_true")

    parser.add_argument("--num-bond-classes", default=5, type=int)
    parser.add_argument("--num-charge-classes", default=6, type=int)

    # BOND PREDICTION AND GUIDANCE:
    parser.add_argument("--bond-guidance-model", default=False, action="store_true")
    parser.add_argument("--bond-prediction", default=False, action="store_true")
    parser.add_argument("--bond-model-guidance", default=False, action="store_true")
    parser.add_argument("--energy-model-guidance", default=False, action="store_true")
    parser.add_argument(
        "--polarizabilty-model-guidance", default=False, action="store_true"
    )
    parser.add_argument("--ckpt-bond-model", default=None, type=str)
    parser.add_argument("--ckpt-energy-model", default=None, type=str)
    parser.add_argument("--ckpt-polarizabilty-model", default=None, type=str)
    parser.add_argument("--guidance-scale", default=1.0e-4, type=float)

    # CONTEXT
    parser.add_argument("--context-mapping", default=False, action="store_true")
    parser.add_argument("--num-context-features", default=0, type=int)
    parser.add_argument("--properties-list", default=[], nargs="+", type=str)

    # PROPERTY PREDICTION
    parser.add_argument("--property-prediction", default=False, action="store_true")

    # LATENT
    parser.add_argument("--prior-beta", default=1.0, type=float)
    parser.add_argument("--sdim-latent", default=256, type=int)
    parser.add_argument("--vdim-latent", default=64, type=int)
    parser.add_argument("--latent-dim", default=None, type=int)
    parser.add_argument("--edim-latent", default=32, type=int)
    parser.add_argument("--num-layers-latent", default=7, type=int)
    parser.add_argument("--latent-layers", default=7, type=int)
    parser.add_argument("--latentmodel", default="diffusion", type=str)
    parser.add_argument("--latent-detach", default=False, action="store_true")

    # GENERAL
    parser.add_argument("-i", "--id", type=int, default=0)
    parser.add_argument("-g", "--gpus", default=1, type=int)
    parser.add_argument("-e", "--num-epochs", default=300, type=int)
    parser.add_argument("--eval-freq", default=1.0, type=float)
    parser.add_argument("--test-interval", default=5, type=int)
    parser.add_argument("-nh", "--no_h", default=False, action="store_true")
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--detect-anomaly", default=False, action="store_true")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument(
        "--max-num-conformers",
        default=5,
        type=int,
        help="Maximum number of conformers per molecule. \
                            Defaults to 30. Set to -1 for all conformers available in database",
    )
    parser.add_argument("--accum-batch", default=1, type=int)
    parser.add_argument("--max-num-neighbors", default=128, type=int)
    parser.add_argument("--ema-decay", default=0.9999, type=float)
    parser.add_argument("--weight-decay", default=0.9999, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--backprop-local", default=False, action="store_true")

    # SAMPLING
    parser.add_argument("--num-test-graphs", default=10000, type=int)
    parser.add_argument("--calculate-energy", default=False, action="store_true")
    parser.add_argument("--save-xyz", default=False, action="store_true")

    return parser
