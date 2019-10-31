from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import pathmagic  # noqa
from panorama.config_gen import PanoramaConfig
from panorama.net.net import PanoramaNet
import random
from panorama.examples.recognition import Recognition

# Change accordingly


class args:
    config_savedir = '../../trained_models/faces_config.json'
    k = 5
    model_save_path = '../../trained_models/panorama_faces_original_loss_weights.h5'
    nms_thr = 0.5
    obj_thr = 0.1
    ver_gamma = 0.9
    save_name = 'panorama_faces_original_loss_ytf_album'
    save_path = '../../trained_models'
    sample_r = 20


def main():
    #  change the dirs accordingly
    all_imgs_dirs = ['.../Images',
                     '.../Images']
    all_anns_dirs = ['.../Annotations',
                     '.../Annotations']
    #  --------------------------------------------------------
    config_gen = PanoramaConfig(args.config_savedir,
                                '',
                                '',
                                '',
                                '',
                                '',
                                args.model_save_path,
                                is_force=False
                                )
    config = config_gen.get_config()
    random.seed(config['random_seed'])
    panoramaNet = PanoramaNet(config)
    panoramaNet.load_weights(args.model_save_path)
    rec = Recognition(panoramaNet, all_imgs_dirs, all_anns_dirs,
                      [], args.obj_thr, args.nms_thr)
    rec.detect_all(args.save_name, args.save_path, sample_r=args.sample_r)


if __name__ == "__main__":
    main()
