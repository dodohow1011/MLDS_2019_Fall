from argparse import ArgumentParser
import os

# this file finds the images used for hw3_2
# i.e. some images with irrelevant labels will be excluded
'''
python3 CreateTrainingTXT.py --anime ../AnimeDataset/tags_clean.csv --extra ../extra_data/tags.csv --content ../TestingTextContent.txt --out ../training.txt
'''

def main(args):
    anime   = args.anime
    extra   = args.extra
    out     = args.out
    content = args.content

    traningtxt = open(out, 'w')
    anime_dir  = '/'.join(anime.split('/')[:-1])
    extra_dir  = '/'.join(extra.split('/')[:-1])
    hair = []
    eyes = []

    with open(content, 'r') as f:
        for line in f:
            line = line.strip()
            if line.endswith('hair'): hair.append(line)
            if line.endswith('eyes'): eyes.append(line)

    with open(anime, 'r') as f:
        for line in f:
            image_path = os.path.join(anime_dir, 'faces64', line.strip().split(',')[0])
            atts = (line.strip().split(',')[-1]).split('\t')
            a_h = None
            a_e = None
            hair_check = False
            eyes_check = False
            for a in atts:
                # check hair attributes
                if not hair_check:
                    for h in hair:
                        if a.startswith(h):
                            a_h = h
                            hair_check = True
                            break
                # check eyes attributes
                if not eyes_check:
                    for e in eyes:
                        if a.startswith(e):
                            a_e = e
                            eyes_check = True
                            break
                if hair_check and eyes_check: break
            if a_h is not None and a_e is not None:
                traningtxt.write('{},{}\n'.format(image_path, ','.join([a_h, a_e])))
                print (image_path)

    with open(extra, 'r') as f:
        for line in f:
            image_path = os.path.join(extra_dir, 'images', line.strip().split(',')[0])
            atts = line.strip().split(',')[-1]
            atts = [atts.split()[:2], atts.split()[2:]]
            atts = [' '.join(atts[0]), ' '.join(atts[1])]
            a_h = None
            a_e = None
            hair_check = False
            eyes_check = False
            for a in atts:
                # check hair attributes
                if not hair_check:
                    for h in hair:
                        if a.startswith(h):
                            a_h = h
                            hair_check = True
                            break
                # check eyes attributes
                if not eyes_check:
                    for e in eyes:
                        if a.startswith(e):
                            a_e = e
                            eyes_check = True
                            break
                if hair_check and eyes_check: break
            if a_h is not None and a_e is not None:
                traningtxt.write('{},{}\n'.format(image_path, ','.join([a_h, a_e])))
                print (image_path)

    traningtxt.close()

def parse():
    parser = ArgumentParser()
    parser.add_argument('--anime',   required=True, help='the tags for AnimeDataset')
    parser.add_argument('--extra',   required=True, help='the tags for extra_data')
    parser.add_argument('--out',     required=True, help='the path to save training.txt')
    parser.add_argument('--content', required=True, help='the file: TestingTextContent')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse())

