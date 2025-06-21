import argparse

def para_parser():
    arg_parser = argparse.ArgumentParser(description='MDvsFA')
    arg_parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    arg_parser.add_argument('--batch-size', type=int, default=10, help='batch size for training')
    arg_parser.add_argument('--parallel', type=bool, default=True, help='use DataParallel if available')
    arg_parser.add_argument('--d-lr', type=float, default=1e-5, help='learning rate for discriminator')
    arg_parser.add_argument('--g1-lr', type=float, default=1e-4, help='learning rate for generator1')
    arg_parser.add_argument('--g2-lr', type=float, default=1e-4, help='learning rate for generator2')
    arg_parser.add_argument('--lambda1', type=float, default=100, help='adversarial loss weight for G1')
    arg_parser.add_argument('--lambda2', type=float, default=1, help='adversarial loss weight for G2')
    arg_parser.add_argument('--training-imgs', default='./data/training/*_1.png', help='glob pattern for training images')
    arg_parser.add_argument('--training-masks', default='./data/training/*_2.png', help='glob pattern for training masks')
    arg_parser.add_argument('--evl-imgs', default='./data/test_org/*', help='glob pattern for eval images')
    arg_parser.add_argument('--evl-masks', default='./data/test_gt/*', help='glob pattern for eval masks')
    arg_parser.add_argument('--is-anm', action='store_true', help='enable anomaly mode if set')
    
    # ─── NEW ARGUMENTS FOR SIRST-DATA AUGMENTATION ───────────────────────────────────
    arg_parser.add_argument(
        '--crop-size',
        type=int,
        default=480,
        help='crop size for synchronized transforms (SIRST loader)'
    )
    arg_parser.add_argument(
        '--base-size',
        type=int,
        default=512,
        help='base resize size for synchronized transforms (SIRST loader)'
    )
    # ────────────────────────────────────────────────────────────────────────────────
    
    args = arg_parser.parse_args()
    return args
