using System;
using Emgu.CV;
using Emgu.CV.Features2D;
using System.IO;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Linq;
using System.Collections.Generic;
using System.Drawing;

namespace FeatureDetection
{
    class Program
    {
        static void Main(string[] args)
        {
            var faceClassifier = new CascadeClassifier(Path.Join("resources",
                "haarcascade_frontalface_default.xml"));
            //var orbDetector = new ORBDetector(10000);
            

            var img = CvInvoke.Imread(Path.Join("resources", "img1.jpg")).ToImage<Bgr,Byte>();
            var img2 = CvInvoke.Imread(Path.Join("resources", "img2.jpg")).ToImage<Bgr, Byte>();
            var img3 = CvInvoke.Imread(Path.Join("resources", "img3.jpg")).ToImage<Bgr, Byte>();
            var cat = CvInvoke.Imread(Path.Join("resources", "catFace.jpg")).ToImage<Bgr, Byte>();
            var dstImg = CvInvoke.Imread(Path.Join("resources", "img1.jpg")).ToImage<Bgr, Byte>();
            var face = faceClassifier.DetectMultiScale(img,
                minSize: new System.Drawing.Size(300, 300))[0];
            var projected = new Mat();
            //System.Drawing.PointF[] srcPoints = { new System.Drawing.PointF( 0,0 ), new System.Drawing.PointF(0, cat.Cols - 1 ), new System.Drawing.PointF{Y= 0, X= cat.Rows - 1 }, new System.Drawing.PointF{Y = cat.Cols - 1, X = cat.Rows - 1 } };
            //System.Drawing.PointF[] dstPoints = { new System.Drawing.PointF{Y= face.Y,X= face.X }, new System.Drawing.PointF{ Y = face.Bottom, X=face.X }, new System.Drawing.PointF{ Y = face.Y, X = face.Right },new System.Drawing.PointF { Y = face.Bottom, X = face.Right } };
            
            var srcPoints = InputImageToPointCorners(cat);
            var dstPoints = FaceToCorners(face);

            var homography = CvInvoke.FindHomography(srcPoints, dstPoints, Emgu.CV.CvEnum.RobustEstimationAlgorithm.Ransac, 5.0);
            CvInvoke.WarpPerspective(cat, projected, homography, img.Size);
            var warpMatrix = CvInvoke.GetAffineTransform(srcPoints, dstPoints);
            img.Mat.CopyTo(projected, 1 - projected);
            //CvInvoke.WarpAffine(cat, img, warpMatrix, img.Size, Emgu.CV.CvEnum.Inter.Linear, Emgu.CV.CvEnum.Warp.Default, Emgu.CV.CvEnum.BorderType.Transparent);




            CvInvoke.Rectangle(dstImg, face,
                    new Emgu.CV.Structure.MCvScalar(255, 0, 0), 10);
            //CvInvoke()
            //CvInvoke.Imshow("Face", dstImg);
            //CvInvoke.Imshow("Projection", projected);
            //CvInvoke.WaitKey(0);
            //img.ROI = face;


            var featuresShown = new Mat();
            var orbDetector = new ORBDetector(5000);
            var features1 = new VectorOfKeyPoint();
            var descriptors1 = new Mat();
            orbDetector.DetectAndCompute(img, null, features1, descriptors1, false);
            Features2DToolbox.DrawKeypoints(img, features1, featuresShown, new Bgr(255, 0, 0));
            //CvInvoke.Imshow("Key points", featuresShown);
            //CvInvoke.WaitKey(0);
            //foreach(var feature in features1.Va)

            var features2 = new VectorOfKeyPoint();
            var descriptors2 = new Mat();
            orbDetector.DetectAndCompute(img2, null, features2, descriptors2, false);

            var knnMatches = new VectorOfVectorOfDMatch();
            
            var matchList = new List<MDMatch>();
            var srcPts = new List<PointF>();
            var dstPts = new List<PointF>();

            var bfMatcher = new BFMatcher(DistanceType.L1);
            bfMatcher.Add(descriptors1);
            bfMatcher.KnnMatch(descriptors2, knnMatches, k:1,mask:null, compactResult:true);
            foreach(var matchSet in knnMatches.ToArrayOfArray())
            {
                if(matchSet.Length>0 && matchSet[0].Distance < 400)
                {
                    matchList.Add(matchSet[0]);
                    var featureModel = features1[matchSet[0].TrainIdx];
                    var featureTrain = features2[matchSet[0].QueryIdx];
                    srcPts.Add(featureModel.Point);
                    dstPts.Add(featureTrain.Point);
                }
            }
            var matches = new VectorOfDMatch(matchList.ToArray());
            var imgOut = new Mat();
            Features2DToolbox.DrawMatches(img, features1, img2, features2, matches, imgOut, new MCvScalar(0, 255, 0), new MCvScalar(255, 0, 0));
            CvInvoke.Imshow("matches", imgOut);
            CvInvoke.WaitKey(0);


            var mtrx = CvInvoke.FindHomography(srcPts.ToArray(), dstPts.ToArray(), Emgu.CV.CvEnum.RobustEstimationAlgorithm.Ransac, 5.0);

            CvInvoke.Multiply(mtrx, homography, homography);
            CvInvoke.WarpPerspective(cat, dstImg, homography, img.Size);

            CvInvoke.Imshow("cat", dstImg);
            CvInvoke.WaitKey(0);

        }

        public static System.Drawing.PointF[] InputImageToPointCorners(Image<Bgr, Byte> img)
        {
            return new System.Drawing.PointF[]{ new System.Drawing.PointF(0, 0), new System.Drawing.PointF(0, img.Cols - 1), new System.Drawing.PointF { Y = 0, X = img.Rows - 1 }, new System.Drawing.PointF { Y = img.Cols - 1, X = img.Rows - 1 } };            
        }

        public static System.Drawing.PointF[] FaceToCorners(System.Drawing.Rectangle face)
        {
            return new System.Drawing.PointF[] { new System.Drawing.PointF { Y = face.Y, X = face.X }, new System.Drawing.PointF { Y = face.Bottom, X = face.X }, new System.Drawing.PointF { Y = face.Y, X = face.Right }, new System.Drawing.PointF { Y = face.Bottom, X = face.Right } };
        }
    }
}
