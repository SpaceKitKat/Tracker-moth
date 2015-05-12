function varargout = test_lensdistort_gui(varargin)
% TEST_LENSDISTORT_GUI MATLAB code for test_lensdistort_gui.fig
%      TEST_LENSDISTORT_GUI, by itself, creates a new TEST_LENSDISTORT_GUI or raises the existing
%      singleton*.
%
%      H = TEST_LENSDISTORT_GUI returns the handle to a new TEST_LENSDISTORT_GUI or the handle to
%      the existing singleton*.
%
%      TEST_LENSDISTORT_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TEST_LENSDISTORT_GUI.M with the given input arguments.
%
%      TEST_LENSDISTORT_GUI('Property','Value',...) creates a new TEST_LENSDISTORT_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before test_lensdistort_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to test_lensdistort_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help test_lensdistort_gui

% Last Modified by GUIDE v2.5 19-Mar-2015 11:33:23

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @test_lensdistort_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @test_lensdistort_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT
end

% --- Executes just before test_lensdistort_gui is made visible.
function test_lensdistort_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to test_lensdistort_gui (see VARARGIN)

% Load frame
x = imread('frame','png');

% Choose default command line output for test_lensdistort_gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% This sets up the initial plot - only do when we are invisible
% so window can get raised using test_lensdistort_gui.
if strcmp(get(hObject,'Visible'),'off')
    plot(rand(5));
end

% UIWAIT makes test_lensdistort_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);
end


% --- Outputs from this function are returned to the command line.
function varargout = test_lensdistort_gui_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
end

% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axes1);
cla;

end

function k_value_Callback(hObject, eventdata, handles)
% hObject    handle to k_value (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of k_value as text
%        str2double(get(hObject,'String')) returns contents of k_value as a double

end

% --- Executes during object creation, after setting all properties.
function k_value_CreateFcn(hObject, eventdata, handles)
% hObject    handle to k_value (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end
