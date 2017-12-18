from time import gmtime, strftime
from keras.callbacks import Callback
import pickle        

class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, name, checkpoint_path, net_name="NET", print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn
        self.checkpoint_path = checkpoint_path
        self.log_file_name = "logs/" +str(name) + ".txt" #_" + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + ".txt")
        log_file = open(self.log_file_name, "a")
        log_file.write("------")
        log_file.write("NEW NET: " + net_name)
        log_file.write("------")
        log_file.close()
        self.best_val_loss = 20

    def on_epoch_end(self, epoch, logs={}):
        msg = "{Epoch: %i} %s \n" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)
        log_file = open(self.log_file_name, "a")
        log_file.write(msg)
        log_file.close()
        """if logs['val_loss'] < self.best_val_loss:
            self.best_val_loss = logs['val_loss']
            print ("Saving momentum for val loss: " + str(logs['val_loss']))
            with open(SAVE_DIR + '{}_mask_4.pickle'.format(subject[0]), 'wb') as handle:
                pickle.dump(self.model.optimizer.get_state(), handle, protocol=pickle.HIGHEST_PROTOCOL)"""


