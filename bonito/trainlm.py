from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os


def main(args):
    workdir = os.path.expanduser(args.training_directory)
    kernel_size = args.kernel_size
    references = args.references_file

    lm_probs = {}

    with open(workdir+'/'+references, 'r') as f:
        f.readline()

        for line in f:
            i = 0

            while i+kernel_size < len(line):
                # read kernel_size characters
                # match kernel_size-tuple agains dictionary
                # if found, add one to count (value)
                # if not found, add to dictionary with count (value) of 1
                # move header one character
                # repeat for all characters in line
                # skip next line as it is an id of a read
                if line[i:i+3] in lm_probs:
                    lm_probs[line[i:i+kernel_size]] += 1
                else:
                    lm_probs[line[i:i+kernel_size]] = 1

                i+=1

            f.readline()

        file = open(workdir+'/'+"language_model.txt", 'w')
        for key, value in lm_probs.items():
            file.write(key.__str__()+';'+value.__str__()+'\n')
        file.close()
        print("Done.")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("training_directory")
    parser.add_argument("references_file")
    parser.add_argument("kernel_size", type=int)

    return parser