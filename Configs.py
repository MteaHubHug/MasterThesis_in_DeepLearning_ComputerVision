class SharedConfigurations():

    def __init__(self):

    # folders

        # training
        self.path_classes= r"G:\Matea\DATASET\usecase Nachverpacken"
        #self.path_classes = r"/home/matea/DATASET/usecase Nachverpacken"  #### training in docker
        self.path_training_results = 'TRAINING/results_nachverpacken'
        self.path_model = 'save_at_180.h5'

        # result preparation
        self.orig_train_folder = r"G:\Matea\proba\Result_nachverpacken\train_dataset"
        self.orig_valid_folder = r"G:\Matea\proba\Result_nachverpacken\validation_dataset"

        #self.orig_train_folder = r"/home/matea/proba/Result_nachverpacken/train_dataset"
        #self.orig_valid_folder = r"/home/matea/proba/Result_nachverpacken/validation_dataset"
    # parameters for training

        self.val_ratio = 0.2
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
        self.string_not_ok_prediction = 'Nachverpacken'

    # results - visualizing :
        self.ok_folder = r"G:\Matea\DATASET\usecase Nachverpacken\OK"
        self.nok_folder = r"G:\Matea\DATASET\usecase Nachverpacken\NOK"

        #self.ok_folder = r"/home/matea/DATASET/usecase Nachverpacken/OK"
        #self.nok_folder = r"/home/matea/DATASET/usecase Nachverpacken/NOK"


    # path to saved .h5 models :
        #self.saved_models=r"G:\Matea\proba\TRAINING\results_nachverpacken\keras_models"

        self.saved_models = r"D:\keras_models" ## on D disk
    # path to augumented examples :
        #self.augumented_examples_path=r"G:\Matea\Augumented_examples"
        self.augumented_examples_path = r"D:\Augumented_examples" # disk D
