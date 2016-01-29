import java.io.File;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;
import org.opencv.objdetect.CascadeClassifier;

public class SimpleSVM {

	private CascadeClassifier faceDetector;
	private Mat image;
	private Mat classes;
	private Mat trainingData;
	private Mat trainingImages;
	private Mat trainingLabels;
	CvSVM svm;


	public void identify(String filename) {
		init();
		detectFace(filename);
		Mat face = Highgui.imread("output.png", Highgui.CV_LOAD_IMAGE_GRAYSCALE);
		identify(face);
	}

	private void init() {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		faceDetector = new CascadeClassifier(new File(
				"src/main/resources/haarcascade_frontalface_alt.xml").getAbsolutePath());
		classes = new Mat();
		trainingData = new Mat();
		trainingImages = new Mat();
		trainingLabels = new Mat();
	}

	private void detectFace(String filename) {
		image = Highgui.imread(filename);
		MatOfRect faceDetections = new MatOfRect();
		faceDetector.detectMultiScale(image, faceDetections);
		System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));
		for (Rect rect : faceDetections.toArray()) {
			Core.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
					new Scalar(0, 255, 0));
		}
		Highgui.imwrite("output.png", image.submat(faceDetections.toArray()[0]));
	}

	private void trainSVM() {
		trainPositive();
		trainNegative();
		trainingImages.copyTo(trainingData);
		trainingData.convertTo(trainingData, CvType.CV_32FC1);
		trainingLabels.copyTo(classes);
	}

	private void trainPositive() {
		Mat img = new Mat();
		Mat con = Highgui.imread("D:\\cybuch\\workspace\\facerecognizer\\src\\main\\resources\\happy.png", Highgui.CV_LOAD_IMAGE_GRAYSCALE);
		con.convertTo(img, CvType.CV_32FC1,1.0/255.0);
		trainingImages.push_back(img.reshape(1, 1));
		trainingLabels.push_back(Mat.ones(new Size(1, 1), CvType.CV_32FC1));
	}

	private void trainNegative() {
		Mat img = new Mat();
		Mat con = Highgui.imread("D:\\cybuch\\workspace\\facerecognizer\\src\\main\\resources\\sad.png", Highgui.CV_LOAD_IMAGE_GRAYSCALE);
		con.convertTo(img, CvType.CV_32FC1,1.0/255.0);
		trainingImages.push_back(img.reshape(1,1));
		trainingLabels.push_back(Mat.zeros(new Size(1, 1), CvType.CV_32FC1));   
	}

	private void identify(Mat face) {
		trainSVM();
		CvSVMParams params = new CvSVMParams();
		params.set_kernel_type(CvSVM.LINEAR);
		svm = new CvSVM(trainingData, classes, new Mat(), new Mat(), params);  
		svm.save("svm.xml");
		svm.load("svm.xml");
		System.out.println(svm);
		Mat out = new Mat();
		face.convertTo(out, CvType.CV_32FC1);
		out.reshape(1, 1);
		System.out.println(out);
		System.out.println(svm.predict(out));
	}
}
