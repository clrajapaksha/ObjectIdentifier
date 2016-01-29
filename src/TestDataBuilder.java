import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.highgui.Highgui;

import libsvm.svm_node;

public class TestDataBuilder {
	
	private String test_path;
	private FeatureDetector detector;
	private DescriptorExtractor extractor;
	private Mat img, descriptors;
	private MatOfKeyPoint keypoints;
	private double[] testData;
	
	public TestDataBuilder(String path){
		test_path = path;
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		//img = Highgui.imread(img_path);
		detector = FeatureDetector.create(FeatureDetector.ORB);
		extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
		keypoints = new MatOfKeyPoint();
		descriptors = new Mat();
		testData = new double[SVMHandler.NumKeyPoints*SVMHandler.DescriptorSize];
	}
	
	public double[] getTestFeatures(String file){
		img = Highgui.imread(test_path+"/"+file,Highgui.CV_LOAD_IMAGE_GRAYSCALE);
		//img.convertTo(img, CvType.CV_32FC1,1.0/255.0);
		detector.detect(img, keypoints);
		extractor.compute(img, keypoints, descriptors);
		//displayImage(Mat2BufferedImage(img));
	    
		for(int row=0;row<SVMHandler.NumKeyPoints;row++){
			for(int col=0; col<SVMHandler.DescriptorSize; col++){
				double pixel = descriptors.get(row, col)[0];
				testData[SVMHandler.NumKeyPoints*row+col] = pixel;
			}
		}
		return testData;
	}
	
	public void overAllTest(){
		int total = SVMHandler.NumClass*SVMHandler.NumTestImgPerClass;
		int correct = 0;
		for(int i=1; i<=SVMHandler.NumClass; i++){
			for(int j=1; j<=SVMHandler.NumTestImgPerClass; j++){
				String file = i+" ("+j+").jpg";
				double response = SVMHandler.test(this.getTestFeatures(file))[0];
				if(i == (int)response)
					correct++;
			}
		}
		System.out.println("Accuracy :"+(double)correct*100/total+"%");
		
	}
}
