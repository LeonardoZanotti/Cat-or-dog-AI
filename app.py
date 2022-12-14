import model
import sys, os
import logging

logger = logging.getLogger('ftpuploader')
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')


class App:

    def __init__(self):
        self.photo = None
        self.model = model.Model()
        self.execute()

    def execute(self):
        try:
            img_path: str = sys.argv[1]
            if not os.path.isfile(img_path):
                logger.error("Incorrect file path!")
                exit(123)
            self.model.train_model()
            prediction: str = self.predict(img_path)
            logger.info(f"It's a {prediction}")
        except IndexError as e:
            logger.error("Inform the image!")
        except Exception as e:
            logger.error(e)

    def predict(self, img: str) -> str:
        return "cat" if self.model.predict(img) == 1 else "dog"

