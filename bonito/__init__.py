import os, sys

modules = ['basecaller', 'evaluate', 'view', 'convert', 'download', 'tune_pbs', 'trainlm']

__version__ = '0.1.5'


def main():
    parser = ArgumentParser(
        'bonito', 
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-v', '--version', action='version',
        version='%(prog)s {}'.format(__version__)
    )

    subparsers = parser.add_subparsers(
        title='subcommands', description='valid commands',
        help='additional help', dest='command'
    )
    subparsers.required = True

    for module in modules:
        mod = globals()[module]
        p = subparsers.add_parser(module, parents=[mod.argparser()])
        p.set_defaults(func=mod.main)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    this_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(this_dir)
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    from bonito import basecaller, convert, download, evaluate, view, tune_pbs, trainlm

    try:
        from bonito import train, tune

        modules.extend(['train', 'tune'])
    except ImportError:
        pass
    main()
