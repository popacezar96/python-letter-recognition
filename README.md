# python-letter-recognition
An Optical Character Recognition (OCR) app written in python3 that uses an artificial intelligence model based on convolutional neural network to classify individual or groups of letters.

The neural network was trained and used to classify one letter at a time. To classify words I used functions from the OpenCV library and some manually written code
to "cut" words into individual contours that were fed to the trained model which returned all the classified letters sequentially. The final result was displayed
to the user in the shape of a selectable string.

To train the model I used tensorflow inside the Jupyter lab IDE.
For graphical processing:
- numpy
- OpenCV

The backend support was provided trough Flask framework.
