class SharedConfigurations():

    def __init__(self):


    # dataset preparation :

        #self.sirius_folder = r"G:\Matea\WUERTH_IRIIS_DISK_COPY\system_OLD"
        #self.iriis_folder = r"G:\Matea\WUERTH_IRIIS_DISK_COPY\system_NEW"

        #self.raw_krdis_folder = r"G:\Matea\krdis1"
        #self.raw_krdis_unique = r"G:\Matea\krdis2"
        #self.converted_krdis=r"G:\Matea\krdis3"

        #self.sirius_folder_krdis_converted=r"G:\Matea\krdis3_converted"

        #self.sirius_dest = r"G:\Matea\FINAL_DATASET\sirius"
        #self.iriis_dest = r"G:\Matea\FINAL_DATASET\iriis"

    # for moving files (move_files.py script) :
        #self.input_file = open('wuerth_iriis.json', 'r')
        #self.NV_OK=r'G:\Matea\DATASET\usecase Nachverpacken\OK'
        #self.NV_NOK=r"G:\Matea\DATASET\usecase Nachverpacken\NOK"
        #self.UM_OK=r'G:\Matea\DATASET\usecase Umschlichten\OK'
        #self.UM_NOK=r"G:\Matea\DATASET\usecase Umschlichten\NOK"


    # training
        self.path_classes = r"D:\DATASET\usecase Umschlichtung"
        #self.path_classes=r"D:\DATASET\usecase Nachverpacken"
        #self.path_classes= r"G:\Matea\DATASET\usecase Nachverpacken"
        #self.path_classes = r"/home/matea/DATASET/usecase Nachverpacken"  #### training in docker
        #self.path_classes = r"/home/matea/usecase Umschlichtung"  #### training in docker




        self.path_training_results = 'TRAINING/results_umschlichtung'
        self.path_model = 'save_at_65.h5'

        # result preparation

        self.orig_train_folder =r"D:\Results\Result_umschlichtung\train_dataset"
        self.orig_valid_folder =r"D:\Results\Result_umschlichtung\validation_dataset"

        #self.orig_train_folder = r"G:\Matea\proba\Result_nachverpacken\train_dataset"
        #self.orig_valid_folder = r"G:\Matea\proba\Result_nachverpacken\validation_dataset"

    # parameters for training

        self.val_ratio = 0.1
        self.img_size = (250, 400)
        self.img_mode = 'rgb'  # 'grayscale'
        self.batch_size = 51
        self.rnd_seed = 48
        self.tot_iterations = 100000
        self.learning_rate=1e-4
    # files for keeping track of image names

        self.train_images_name_file = 'train_file_names.txt'
        self.validation_images_name_file = 'valid_file_names.txt'
    # result preparation folders

        self.string_ok_prediction = 'OK'
        self.string_not_ok_prediction = 'Umschlichtung'

    # results - visualizing :
        self.ok_folder = r"D:\DATASET\usecase Umschlichtung\OK"
        self.nok_folder = r"D:\DATASET\usecase Umschlichtung\NOK"
        #self.ok_folder = r"G:\Matea\DATASET\usecase Nachverpacken\OK"
        #self.nok_folder = r"G:\Matea\DATASET\usecase Nachverpacken\NOK"

        #self.ok_folder = r"/home/matea/DATASET/usecase Nachverpacken/OK"
        #self.nok_folder = r"/home/matea/DATASET/usecase Nachverpacken/NOK"


    # path to saved .h5 models :
        #self.saved_models=r"G:\Matea\proba\TRAINING\results_nachverpacken\keras_models"

        #self.saved_models = r"D:\keras_models" ## on D disk
    # path to augumented examples :
        #self.augumented_examples_path=r"G:\Matea\Augumented_examples"
        #self.augumented_examples_path = r"D:\Augumented_examples" # disk D
############################################################################################
########################### CORNER DETECTOR STUFF : ########################################

    # IRIIS and SIRIUS json files - for corner detector (box corners annotations)
        self.IRIIS_json = open('IRIIS Dataset.json', 'r')
        self.SIRIUS_json = open('SIRIUS Dataset.json', 'r')
        self.annotated_IRIIS_images_folder= r"D:\FINAL DATASET\wuerth_iriis_annotate"
        self.annotated_SIRIUS_images_folder = r"D:\FINAL DATASET\wuerth_sirius_annotate"

        self.not_annotated_IRIIS_images_folder= r"C:\wuerth_iriis_theRest"
        self.not_annotated_SIRIUS_images_folder = r"D:\FINAL DATASET\wuerth_sirius_theRest"
        self.keypoint_detector_models_path=r"C:\Keypoint_detec_models"
        self.keypoint_detec_model = "keypoint_detector_model.h5"
        self.keypoint_detec_results_path= r"C:\Keypoint_detec_results"

        self.keypoint_detec_IMG_SIZE = 224
        self.keypoint_detec_BATCH_SIZE = 64
        self.keypoint_detec_EPOCHS = 3000
        self.num_keypoints= 4 * 2  # 4 pairs each having x and y coordinates






