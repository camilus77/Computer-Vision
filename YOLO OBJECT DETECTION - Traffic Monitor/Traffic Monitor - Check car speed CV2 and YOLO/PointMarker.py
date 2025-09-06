import cv2 as cv

class PointMarker:
    """Mark points on an image using Left Mouse Button click."""

    def __init__(self, window = "Image"):
        self._window = window
        self._points= list()

    def __call__(self, image, inplace = False):
        return self.mark(image, inplace)

    @property
    def points(self):
        return self._points

    def mark(self, image, inplace = False):
        if not inplace:
            image = image.copy()
        cv.namedWindow(self._window, cv.WINDOW_NORMAL)
        cv.setMouseCallback(self._window, self._record_point, param=image)

        while True:
            cv.imshow(self._window, image)
            if cv.waitKey(1) == ord("q"):
                break

        cv.destroyAllWindows()
        return self._points

    def _record_point(self, event, x, y, flags, image):
        if event == cv.EVENT_LBUTTONDOWN:
            self._points.append((x, y))
            if image is not None:
                self._draw_point(image, (x, y))

    def _draw_point(self, image, point):
        cv.drawMarker(image, point, (0, 123, 255), cv.MARKER_CROSS, 20, 4, cv.LINE_AA)
