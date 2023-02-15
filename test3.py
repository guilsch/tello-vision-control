import sys

from handGestureModel.model import KeyPointClassifier

sys.path.append('handGestureModel')

classifier = KeyPointClassifier()

print(classifier)