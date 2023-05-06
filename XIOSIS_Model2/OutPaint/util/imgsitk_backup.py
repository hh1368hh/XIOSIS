import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
# from util import registration_utilities as ru
# from util import registration_callbacks as rc
import util.registration_utilities as ru
import util.registration_callbacks as rc


def ResizePatient(original_CT):

    #original_CT = patient_CT #sitk.ReadImage(patient_CT,sitk.sitkInt32)
    dimension = original_CT.GetDimension()
    size=original_CT.GetSize()
    spacing=original_CT.GetSpacing()

    reference_physical_size = np.zeros(original_CT.GetDimension())
    # print(reference_physical_size)
    reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(original_CT.GetSize(), original_CT.GetSpacing(), reference_physical_size)]

    reference_origin = original_CT.GetOrigin()
    reference_direction = original_CT.GetDirection()

    reference_size = tuple(round(ele1 * ele2) for ele1, ele2 in zip(size, spacing)) #[round(sz/resize_factor) for sz in original_CT.GetSize()] 
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

    reference_image = sitk.Image(reference_size, original_CT.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    # reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    
    transform = sitk.AffineTransform(dimension)
    # transform.SetMatrix(original_CT.GetDirection())

    transform.SetTranslation(np.array(original_CT.GetOrigin()) - reference_origin)
  
    # centering_transform = sitk.TranslationTransform(dimension)
    # img_center = np.array(original_CT.TransformContinuousIndexToPhysicalPoint(np.array(original_CT.GetSize())/2.0))
    # centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    # centered_transform.AddTransform(centering_transform)

    # sitk.Show(sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0))
    out=sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, -1000)
    out=copy_metadata(original_CT,out)
    return out



def display_images_with_alpha(image_z, alpha, fixed, moving):
    img = (1.0 - alpha)*fixed[:,:,image_z] + alpha*moving[:,:,image_z] 
    plt.figure(figsize=(16,9))
    plt.imshow(sitk.GetArrayViewFromImage(img),cmap=plt.cm.Greys_r);
    plt.axis('off')
    plt.show()



def copy_metadata(input,out):
    
    for k in input.GetMetaDataKeys():
        v = input.GetMetaData(k)
        out.SetMetaData(k,v)
    return out



def couch_remove(input):

    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    seg = otsu_filter.Execute(input)
    # sitk.Show(seg)
    CCfilter=sitk.ConnectedComponentImageFilter()
    ccf=CCfilter.Execute(seg)
    
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.ConnectedComponent(ccf))

    label_sizes = [ stats.GetNumberOfPixels(l) for l in stats.GetLabels()]

    cci=label_sizes.index(max(label_sizes))

    skin=ccf==cci+1
    # sitk.Show(skin)
    for i in range(0,4):
        filt=sitk.BinaryContourImageFilter()
        skin_contour = filt.Execute(skin)

        skin=skin-skin_contour
    # sitk.Show(skin_contour)
    # sitk.Show(skin)

    filt=sitk.BinaryContourImageFilter()
    skin_contour = filt.Execute(skin)
    s1=skin*1000
    # sitk.Show(s1)
    filt=sitk.ConnectedThresholdImageFilter()
    filt.AddSeed([0,0,0])
    # filt.SetLower=0
    # filt.SetUpper=0.5
    skin_out = filt.Execute(s1)
    # sitk.Show(skin_out)

    filt=sitk.NotImageFilter()
    skin=filt.Execute(skin_out)
    # sitk.Show(skin)
    # filt=sitk.BinaryMorphologicalOpeningImageFilter()
    # skin=filt.Execute(skin)

    # filt=sitk.BinaryFillholeImageFilter()
    # skin=filt.Execute(skin)

    # filt=sitk.VotingBinaryIterativeHoleFillingImageFilter()
    # filt.SetRadius(10)
    # filt.SetMajorityThreshold(1)
    # filt.SetBackgroundValue(0)
    # filt.SetForegroundValue(1)
    # skin=filt.Execute(skin)

    moving_np=sitk.GetArrayFromImage(input)
    skin_np=sitk.GetArrayFromImage(skin)
    background=np.ones(moving_np.shape) * -1000
    background[skin_np==1]=moving_np[skin_np==1]
    moving_skin=sitk.GetImageFromArray(background)
    moving_skin.CopyInformation(input)

    return copy_metadata(input,moving_skin), skin

def bspline_intra_modal_registration(fixed_image, moving_image, fixed_image_mask=None, fixed_points=None, moving_points=None):

    registration_method = sitk.ImageRegistrationMethod()
    
    # Determine the number of BSpline control points using the physical spacing we want for the control grid. 
    grid_physical_spacing = [50.0, 50.0, 50.0] # A control point every 50mm
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]

    initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, 
                                                         transformDomainMeshSize = mesh_size, order=3)    
    registration_method.SetInitialTransform(initial_transform)
        
    registration_method.SetMetricAsMeanSquares()
    # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be 
    # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the
    # whole image.
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    if fixed_image_mask:
        registration_method.SetMetricFixedMask(fixed_image_mask)
    
    # Multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)
    

    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    if fixed_points and moving_points:
        registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rc.metric_and_reference_plot_values(registration_method, fixed_points, moving_points))

    else:
        registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
        registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    
    return registration_method.Execute(fixed_image, moving_image)



def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

def plot_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))



def rigid_registration(fixed_NC, moving_NC,initial_transform = None):

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    if  initial_transform is not None:
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    final_transform = registration_method.Execute(fixed_NC, moving_NC)

    return final_transform



def hu_normalize(image,max=1200,min=-1000):

    resacleFilter = sitk.IntensityWindowingImageFilter()
    resacleFilter.SetWindowMaximum(1200)
    resacleFilter.SetWindowMinimum(-1000)
    resacleFilter.SetOutputMaximum(1200)
    resacleFilter.SetOutputMinimum(-1000)

    image = resacleFilter.Execute(image)
    return image