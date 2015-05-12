k_distortion = -0.45; % use -1 < k < 0 to undistort fish eye effect 
x = imread('frame','png');
y = lensdistort(x,k_distortion,'bordertype','crop','ftype',3); 
% y = lensdistort(x,k_distortion,'ftype',2); 

figure, subplot(121), imshow(x),title('original');
subplot(122), imshow(y),...
    title(sprintf('undistorted with k=%.2f',k_distortion));