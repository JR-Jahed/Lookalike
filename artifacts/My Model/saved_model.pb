╤Ъ
╩Щ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

√
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Т
ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58ню
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
{
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╞
*"
shared_nameAdam/v/dense/bias
t
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes	
:╞
*
dtype0
{
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╞
*"
shared_nameAdam/m/dense/bias
t
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes	
:╞
*
dtype0
Д
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А╞
*$
shared_nameAdam/v/dense/kernel
}
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel* 
_output_shapes
:
А╞
*
dtype0
Д
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А╞
*$
shared_nameAdam/m/dense/kernel
}
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel* 
_output_shapes
:
А╞
*
dtype0
}
Adam/v/dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*#
shared_nameAdam/v/dense1/bias
v
&Adam/v/dense1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense1/bias*
_output_shapes	
:А*
dtype0
}
Adam/m/dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*#
shared_nameAdam/m/dense1/bias
v
&Adam/m/dense1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense1/bias*
_output_shapes	
:А*
dtype0
Ж
Adam/v/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*%
shared_nameAdam/v/dense1/kernel

(Adam/v/dense1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense1/kernel* 
_output_shapes
:
АА*
dtype0
Ж
Adam/m/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*%
shared_nameAdam/m/dense1/kernel

(Adam/m/dense1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense1/kernel* 
_output_shapes
:
АА*
dtype0
Ы
!Adam/v/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/v/batch_normalization_4/beta
Ф
5Adam/v/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_4/beta*
_output_shapes	
:А*
dtype0
Ы
!Adam/m/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/m/batch_normalization_4/beta
Ф
5Adam/m/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_4/beta*
_output_shapes	
:А*
dtype0
Э
"Adam/v/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/v/batch_normalization_4/gamma
Ц
6Adam/v/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_4/gamma*
_output_shapes	
:А*
dtype0
Э
"Adam/m/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/m/batch_normalization_4/gamma
Ц
6Adam/m/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_4/gamma*
_output_shapes	
:А*
dtype0
{
Adam/v/conv5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/v/conv5/bias
t
%Adam/v/conv5/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv5/bias*
_output_shapes	
:А*
dtype0
{
Adam/m/conv5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/m/conv5/bias
t
%Adam/m/conv5/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv5/bias*
_output_shapes	
:А*
dtype0
Л
Adam/v/conv5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*$
shared_nameAdam/v/conv5/kernel
Д
'Adam/v/conv5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv5/kernel*'
_output_shapes
:@А*
dtype0
Л
Adam/m/conv5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*$
shared_nameAdam/m/conv5/kernel
Д
'Adam/m/conv5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv5/kernel*'
_output_shapes
:@А*
dtype0
Ъ
!Adam/v/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/v/batch_normalization_3/beta
У
5Adam/v/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_3/beta*
_output_shapes
:@*
dtype0
Ъ
!Adam/m/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/m/batch_normalization_3/beta
У
5Adam/m/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_3/beta*
_output_shapes
:@*
dtype0
Ь
"Adam/v/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/v/batch_normalization_3/gamma
Х
6Adam/v/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_3/gamma*
_output_shapes
:@*
dtype0
Ь
"Adam/m/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/m/batch_normalization_3/gamma
Х
6Adam/m/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_3/gamma*
_output_shapes
:@*
dtype0
z
Adam/v/conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/v/conv4/bias
s
%Adam/v/conv4/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv4/bias*
_output_shapes
:@*
dtype0
z
Adam/m/conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/m/conv4/bias
s
%Adam/m/conv4/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv4/bias*
_output_shapes
:@*
dtype0
К
Adam/v/conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameAdam/v/conv4/kernel
Г
'Adam/v/conv4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv4/kernel*&
_output_shapes
:@@*
dtype0
К
Adam/m/conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameAdam/m/conv4/kernel
Г
'Adam/m/conv4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv4/kernel*&
_output_shapes
:@@*
dtype0
Ъ
!Adam/v/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/v/batch_normalization_2/beta
У
5Adam/v/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_2/beta*
_output_shapes
:@*
dtype0
Ъ
!Adam/m/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/m/batch_normalization_2/beta
У
5Adam/m/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_2/beta*
_output_shapes
:@*
dtype0
Ь
"Adam/v/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/v/batch_normalization_2/gamma
Х
6Adam/v/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_2/gamma*
_output_shapes
:@*
dtype0
Ь
"Adam/m/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/m/batch_normalization_2/gamma
Х
6Adam/m/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_2/gamma*
_output_shapes
:@*
dtype0
z
Adam/v/conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/v/conv3/bias
s
%Adam/v/conv3/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3/bias*
_output_shapes
:@*
dtype0
z
Adam/m/conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/m/conv3/bias
s
%Adam/m/conv3/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3/bias*
_output_shapes
:@*
dtype0
К
Adam/v/conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*$
shared_nameAdam/v/conv3/kernel
Г
'Adam/v/conv3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3/kernel*&
_output_shapes
: @*
dtype0
К
Adam/m/conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*$
shared_nameAdam/m/conv3/kernel
Г
'Adam/m/conv3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3/kernel*&
_output_shapes
: @*
dtype0
Ъ
!Adam/v/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/v/batch_normalization_1/beta
У
5Adam/v/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_1/beta*
_output_shapes
: *
dtype0
Ъ
!Adam/m/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/m/batch_normalization_1/beta
У
5Adam/m/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_1/beta*
_output_shapes
: *
dtype0
Ь
"Adam/v/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/v/batch_normalization_1/gamma
Х
6Adam/v/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_1/gamma*
_output_shapes
: *
dtype0
Ь
"Adam/m/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/m/batch_normalization_1/gamma
Х
6Adam/m/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_1/gamma*
_output_shapes
: *
dtype0
z
Adam/v/conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/v/conv2/bias
s
%Adam/v/conv2/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2/bias*
_output_shapes
: *
dtype0
z
Adam/m/conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/m/conv2/bias
s
%Adam/m/conv2/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2/bias*
_output_shapes
: *
dtype0
К
Adam/v/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *$
shared_nameAdam/v/conv2/kernel
Г
'Adam/v/conv2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2/kernel*&
_output_shapes
:  *
dtype0
К
Adam/m/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *$
shared_nameAdam/m/conv2/kernel
Г
'Adam/m/conv2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2/kernel*&
_output_shapes
:  *
dtype0
Ц
Adam/v/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/v/batch_normalization/beta
П
3Adam/v/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/v/batch_normalization/beta*
_output_shapes
: *
dtype0
Ц
Adam/m/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/m/batch_normalization/beta
П
3Adam/m/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/m/batch_normalization/beta*
_output_shapes
: *
dtype0
Ш
 Adam/v/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/v/batch_normalization/gamma
С
4Adam/v/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/v/batch_normalization/gamma*
_output_shapes
: *
dtype0
Ш
 Adam/m/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/m/batch_normalization/gamma
С
4Adam/m/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/m/batch_normalization/gamma*
_output_shapes
: *
dtype0
z
Adam/v/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/v/conv1/bias
s
%Adam/v/conv1/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv1/bias*
_output_shapes
: *
dtype0
z
Adam/m/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/m/conv1/bias
s
%Adam/m/conv1/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv1/bias*
_output_shapes
: *
dtype0
К
Adam/v/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/v/conv1/kernel
Г
'Adam/v/conv1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv1/kernel*&
_output_shapes
: *
dtype0
К
Adam/m/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/m/conv1/kernel
Г
'Adam/m/conv1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv1/kernel*&
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╞
*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:╞
*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А╞
*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
А╞
*
dtype0
o
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense1/bias
h
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes	
:А*
dtype0
x
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namedense1/kernel
q
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel* 
_output_shapes
:
АА*
dtype0
г
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_4/moving_variance
Ь
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_4/moving_mean
Ф
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes	
:А*
dtype0
Н
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_4/beta
Ж
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes	
:А*
dtype0
П
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_4/gamma
И
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes	
:А*
dtype0
m

conv5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name
conv5/bias
f
conv5/bias/Read/ReadVariableOpReadVariableOp
conv5/bias*
_output_shapes	
:А*
dtype0
}
conv5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*
shared_nameconv5/kernel
v
 conv5/kernel/Read/ReadVariableOpReadVariableOpconv5/kernel*'
_output_shapes
:@А*
dtype0
в
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_3/moving_variance
Ы
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_3/beta
Е
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_3/gamma
З
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:@*
dtype0
l

conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
conv4/bias
e
conv4/bias/Read/ReadVariableOpReadVariableOp
conv4/bias*
_output_shapes
:@*
dtype0
|
conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_nameconv4/kernel
u
 conv4/kernel/Read/ReadVariableOpReadVariableOpconv4/kernel*&
_output_shapes
:@@*
dtype0
в
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
l

conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
conv3/bias
e
conv3/bias/Read/ReadVariableOpReadVariableOp
conv3/bias*
_output_shapes
:@*
dtype0
|
conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameconv3/kernel
u
 conv3/kernel/Read/ReadVariableOpReadVariableOpconv3/kernel*&
_output_shapes
: @*
dtype0
в
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_1/moving_variance
Ы
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_1/moving_mean
У
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
М
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_1/beta
Е
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
: *
dtype0
О
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_1/gamma
З
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
: *
dtype0
l

conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
conv2/bias
e
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
_output_shapes
: *
dtype0
|
conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_nameconv2/kernel
u
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*&
_output_shapes
:  *
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0
l

conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
conv1/bias
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes
: *
dtype0
|
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
: *
dtype0
Ч
 serving_default_sequential_inputPlaceholder*1
_output_shapes
:         ЦЦ*
dtype0*&
shape:         ЦЦ
╛	
StatefulPartitionedCallStatefulPartitionedCall serving_default_sequential_inputconv1/kernel
conv1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2/kernel
conv2/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv3/kernel
conv3/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv4/kernel
conv4/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv5/kernel
conv5/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense1/kerneldense1/biasdense/kernel
dense/bias*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╞
*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_44386

NoOpNoOp
Ў┌
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*░┌
valueе┌Bб┌ BЩ┌
ш
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
layer-16
layer-17
layer_with_weights-8
layer-18
layer_with_weights-9
layer-19
layer-20
layer-21
layer-22
layer_with_weights-10
layer-23
layer-24
layer_with_weights-11
layer-25
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!_default_save_signature
"	optimizer
#
signatures*
и
$layer-0
%layer-1
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses* 
* 
╚
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op*
╒
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;axis
	<gamma
=beta
>moving_mean
?moving_variance*
О
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses* 
О
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses* 
╚
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses

Rkernel
Sbias
 T_jit_compiled_convolution_op*
╒
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[axis
	\gamma
]beta
^moving_mean
_moving_variance*
О
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses* 
О
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses* 
╚
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

rkernel
sbias
 t_jit_compiled_convolution_op*
╒
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
{axis
	|gamma
}beta
~moving_mean
moving_variance*
Ф
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses* 
Ф
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
К__call__
+Л&call_and_return_all_conditional_losses* 
╤
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses
Тkernel
	Уbias
!Ф_jit_compiled_convolution_op*
р
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses
	Ыaxis

Ьgamma
	Эbeta
Юmoving_mean
Яmoving_variance*
Ф
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses* 
Ф
ж	variables
зtrainable_variables
иregularization_losses
й	keras_api
к__call__
+л&call_and_return_all_conditional_losses* 
╤
м	variables
нtrainable_variables
оregularization_losses
п	keras_api
░__call__
+▒&call_and_return_all_conditional_losses
▓kernel
	│bias
!┤_jit_compiled_convolution_op*
р
╡	variables
╢trainable_variables
╖regularization_losses
╕	keras_api
╣__call__
+║&call_and_return_all_conditional_losses
	╗axis

╝gamma
	╜beta
╛moving_mean
┐moving_variance*
Ф
└	variables
┴trainable_variables
┬regularization_losses
├	keras_api
─__call__
+┼&call_and_return_all_conditional_losses* 
Ф
╞	variables
╟trainable_variables
╚regularization_losses
╔	keras_api
╩__call__
+╦&call_and_return_all_conditional_losses* 
Ф
╠	variables
═trainable_variables
╬regularization_losses
╧	keras_api
╨__call__
+╤&call_and_return_all_conditional_losses* 
о
╥	variables
╙trainable_variables
╘regularization_losses
╒	keras_api
╓__call__
+╫&call_and_return_all_conditional_losses
╪kernel
	┘bias*
м
┌	variables
█trainable_variables
▄regularization_losses
▌	keras_api
▐__call__
+▀&call_and_return_all_conditional_losses
р_random_generator* 
о
с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
х__call__
+ц&call_and_return_all_conditional_losses
чkernel
	шbias*
Ъ
20
31
<2
=3
>4
?5
R6
S7
\8
]9
^10
_11
r12
s13
|14
}15
~16
17
Т18
У19
Ь20
Э21
Ю22
Я23
▓24
│25
╝26
╜27
╛28
┐29
╪30
┘31
ч32
ш33*
╞
20
31
<2
=3
R4
S5
\6
]7
r8
s9
|10
}11
Т12
У13
Ь14
Э15
▓16
│17
╝18
╜19
╪20
┘21
ч22
ш23*
* 
╡
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
!_default_save_signature
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*
:
юtrace_0
яtrace_1
Ёtrace_2
ёtrace_3* 
:
Єtrace_0
єtrace_1
Їtrace_2
їtrace_3* 
* 
И
Ў
_variables
ў_iterations
°_learning_rate
∙_index_dict
·
_momentums
√_velocities
№_update_step_xla*

¤serving_default* 
Ф
■	variables
 trainable_variables
Аregularization_losses
Б	keras_api
В__call__
+Г&call_and_return_all_conditional_losses* 
Ф
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
И__call__
+Й&call_and_return_all_conditional_losses* 
* 
* 
* 
Ц
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses* 
:
Пtrace_0
Рtrace_1
Сtrace_2
Тtrace_3* 
:
Уtrace_0
Фtrace_1
Хtrace_2
Цtrace_3* 

20
31*

20
31*
* 
Ш
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

Ьtrace_0* 

Эtrace_0* 
\V
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
<0
=1
>2
?3*

<0
=1*
* 
Ш
Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

гtrace_0
дtrace_1* 

еtrace_0
жtrace_1* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
зnon_trainable_variables
иlayers
йmetrics
 кlayer_regularization_losses
лlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses* 

мtrace_0* 

нtrace_0* 
* 
* 
* 
Ц
оnon_trainable_variables
пlayers
░metrics
 ▒layer_regularization_losses
▓layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses* 

│trace_0* 

┤trace_0* 

R0
S1*

R0
S1*
* 
Ш
╡non_trainable_variables
╢layers
╖metrics
 ╕layer_regularization_losses
╣layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

║trace_0* 

╗trace_0* 
\V
VARIABLE_VALUEconv2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
\0
]1
^2
_3*

\0
]1*
* 
Ш
╝non_trainable_variables
╜layers
╛metrics
 ┐layer_regularization_losses
└layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

┴trace_0
┬trace_1* 

├trace_0
─trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
┼non_trainable_variables
╞layers
╟metrics
 ╚layer_regularization_losses
╔layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses* 

╩trace_0* 

╦trace_0* 
* 
* 
* 
Ц
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses* 

╤trace_0* 

╥trace_0* 

r0
s1*

r0
s1*
* 
Ш
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses*

╪trace_0* 

┘trace_0* 
\V
VARIABLE_VALUEconv3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
|0
}1
~2
3*

|0
}1*
* 
Ш
┌non_trainable_variables
█layers
▄metrics
 ▌layer_regularization_losses
▐layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*

▀trace_0
рtrace_1* 

сtrace_0
тtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
А	variables
Бtrainable_variables
Вregularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses* 

шtrace_0* 

щtrace_0* 
* 
* 
* 
Ь
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
Ж	variables
Зtrainable_variables
Иregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses* 

яtrace_0* 

Ёtrace_0* 

Т0
У1*

Т0
У1*
* 
Ю
ёnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses*

Ўtrace_0* 

ўtrace_0* 
\V
VARIABLE_VALUEconv4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
Ь0
Э1
Ю2
Я3*

Ь0
Э1*
* 
Ю
°non_trainable_variables
∙layers
·metrics
 √layer_regularization_losses
№layer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses*

¤trace_0
■trace_1* 

 trace_0
Аtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses* 

Жtrace_0* 

Зtrace_0* 
* 
* 
* 
Ь
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
ж	variables
зtrainable_variables
иregularization_losses
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses* 

Нtrace_0* 

Оtrace_0* 

▓0
│1*

▓0
│1*
* 
Ю
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
м	variables
нtrainable_variables
оregularization_losses
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses*

Фtrace_0* 

Хtrace_0* 
\V
VARIABLE_VALUEconv5/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
╝0
╜1
╛2
┐3*

╝0
╜1*
* 
Ю
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
╡	variables
╢trainable_variables
╖regularization_losses
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses*

Ыtrace_0
Ьtrace_1* 

Эtrace_0
Юtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
└	variables
┴trainable_variables
┬regularization_losses
─__call__
+┼&call_and_return_all_conditional_losses
'┼"call_and_return_conditional_losses* 

дtrace_0* 

еtrace_0* 
* 
* 
* 
Ь
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
╞	variables
╟trainable_variables
╚regularization_losses
╩__call__
+╦&call_and_return_all_conditional_losses
'╦"call_and_return_conditional_losses* 

лtrace_0* 

мtrace_0* 
* 
* 
* 
Ь
нnon_trainable_variables
оlayers
пmetrics
 ░layer_regularization_losses
▒layer_metrics
╠	variables
═trainable_variables
╬regularization_losses
╨__call__
+╤&call_and_return_all_conditional_losses
'╤"call_and_return_conditional_losses* 

▓trace_0* 

│trace_0* 

╪0
┘1*

╪0
┘1*
* 
Ю
┤non_trainable_variables
╡layers
╢metrics
 ╖layer_regularization_losses
╕layer_metrics
╥	variables
╙trainable_variables
╘regularization_losses
╓__call__
+╫&call_and_return_all_conditional_losses
'╫"call_and_return_conditional_losses*

╣trace_0* 

║trace_0* 
^X
VARIABLE_VALUEdense1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
╗non_trainable_variables
╝layers
╜metrics
 ╛layer_regularization_losses
┐layer_metrics
┌	variables
█trainable_variables
▄regularization_losses
▐__call__
+▀&call_and_return_all_conditional_losses
'▀"call_and_return_conditional_losses* 

└trace_0
┴trace_1* 

┬trace_0
├trace_1* 
* 

ч0
ш1*

ч0
ш1*
* 
Ю
─non_trainable_variables
┼layers
╞metrics
 ╟layer_regularization_losses
╚layer_metrics
с	variables
тtrainable_variables
уregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses*

╔trace_0* 

╩trace_0* 
]W
VARIABLE_VALUEdense/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUE
dense/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
N
>0
?1
^2
_3
~4
5
Ю6
Я7
╛8
┐9*
╩
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25*

╦0
╠1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
│
ў0
═1
╬2
╧3
╨4
╤5
╥6
╙7
╘8
╒9
╓10
╫11
╪12
┘13
┌14
█15
▄16
▌17
▐18
▀19
р20
с21
т22
у23
ф24
х25
ц26
ч27
ш28
щ29
ъ30
ы31
ь32
э33
ю34
я35
Ё36
ё37
Є38
є39
Ї40
ї41
Ў42
ў43
°44
∙45
·46
√47
№48*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
╥
═0
╧1
╤2
╙3
╒4
╫5
┘6
█7
▌8
▀9
с10
у11
х12
ч13
щ14
ы15
э16
я17
ё18
є19
ї20
ў21
∙22
√23*
╥
╬0
╨1
╥2
╘3
╓4
╪5
┌6
▄7
▐8
р9
т10
ф11
ц12
ш13
ъ14
ь15
ю16
Ё17
Є18
Ї19
Ў20
°21
·22
№23*
* 
* 
* 
* 
* 
Ь
¤non_trainable_variables
■layers
 metrics
 Аlayer_regularization_losses
Бlayer_metrics
■	variables
 trainable_variables
Аregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses* 

Вtrace_0* 

Гtrace_0* 
* 
* 
* 
Ь
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses* 

Йtrace_0* 

Кtrace_0* 
* 

$0
%1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

>0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

^0
_1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

~0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Ю0
Я1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

╛0
┐1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
Л	variables
М	keras_api

Нtotal

Оcount*
M
П	variables
Р	keras_api

Сtotal

Тcount
У
_fn_kwargs*
^X
VARIABLE_VALUEAdam/m/conv1/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/conv1/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/conv1/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/conv1/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/batch_normalization/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/batch_normalization/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/batch_normalization/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/batch_normalization/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/conv2/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv2/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/conv2/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/conv2/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_1/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_1/gamma2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_1/beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_1/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv3/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv3/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/conv3/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/conv3/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_2/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_2/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_2/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_2/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv4/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv4/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/conv4/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/conv4/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_3/gamma2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_3/gamma2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_3/beta2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_3/beta2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv5/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv5/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/conv5/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/conv5/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_4/gamma2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_4/gamma2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_4/beta2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_4/beta2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense1/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense1/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense1/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense1/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense/kernel2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense/kernel2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/dense/bias2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/dense/bias2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Н0
О1*

Л	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

С0
Т1*

П	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╜!
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp conv3/kernel/Read/ReadVariableOpconv3/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp conv4/kernel/Read/ReadVariableOpconv4/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp conv5/kernel/Read/ReadVariableOpconv5/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp'Adam/m/conv1/kernel/Read/ReadVariableOp'Adam/v/conv1/kernel/Read/ReadVariableOp%Adam/m/conv1/bias/Read/ReadVariableOp%Adam/v/conv1/bias/Read/ReadVariableOp4Adam/m/batch_normalization/gamma/Read/ReadVariableOp4Adam/v/batch_normalization/gamma/Read/ReadVariableOp3Adam/m/batch_normalization/beta/Read/ReadVariableOp3Adam/v/batch_normalization/beta/Read/ReadVariableOp'Adam/m/conv2/kernel/Read/ReadVariableOp'Adam/v/conv2/kernel/Read/ReadVariableOp%Adam/m/conv2/bias/Read/ReadVariableOp%Adam/v/conv2/bias/Read/ReadVariableOp6Adam/m/batch_normalization_1/gamma/Read/ReadVariableOp6Adam/v/batch_normalization_1/gamma/Read/ReadVariableOp5Adam/m/batch_normalization_1/beta/Read/ReadVariableOp5Adam/v/batch_normalization_1/beta/Read/ReadVariableOp'Adam/m/conv3/kernel/Read/ReadVariableOp'Adam/v/conv3/kernel/Read/ReadVariableOp%Adam/m/conv3/bias/Read/ReadVariableOp%Adam/v/conv3/bias/Read/ReadVariableOp6Adam/m/batch_normalization_2/gamma/Read/ReadVariableOp6Adam/v/batch_normalization_2/gamma/Read/ReadVariableOp5Adam/m/batch_normalization_2/beta/Read/ReadVariableOp5Adam/v/batch_normalization_2/beta/Read/ReadVariableOp'Adam/m/conv4/kernel/Read/ReadVariableOp'Adam/v/conv4/kernel/Read/ReadVariableOp%Adam/m/conv4/bias/Read/ReadVariableOp%Adam/v/conv4/bias/Read/ReadVariableOp6Adam/m/batch_normalization_3/gamma/Read/ReadVariableOp6Adam/v/batch_normalization_3/gamma/Read/ReadVariableOp5Adam/m/batch_normalization_3/beta/Read/ReadVariableOp5Adam/v/batch_normalization_3/beta/Read/ReadVariableOp'Adam/m/conv5/kernel/Read/ReadVariableOp'Adam/v/conv5/kernel/Read/ReadVariableOp%Adam/m/conv5/bias/Read/ReadVariableOp%Adam/v/conv5/bias/Read/ReadVariableOp6Adam/m/batch_normalization_4/gamma/Read/ReadVariableOp6Adam/v/batch_normalization_4/gamma/Read/ReadVariableOp5Adam/m/batch_normalization_4/beta/Read/ReadVariableOp5Adam/v/batch_normalization_4/beta/Read/ReadVariableOp(Adam/m/dense1/kernel/Read/ReadVariableOp(Adam/v/dense1/kernel/Read/ReadVariableOp&Adam/m/dense1/bias/Read/ReadVariableOp&Adam/v/dense1/bias/Read/ReadVariableOp'Adam/m/dense/kernel/Read/ReadVariableOp'Adam/v/dense/kernel/Read/ReadVariableOp%Adam/m/dense/bias/Read/ReadVariableOp%Adam/v/dense/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*e
Tin^
\2Z	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_45737
╪
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2/kernel
conv2/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv3/kernel
conv3/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv4/kernel
conv4/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv5/kernel
conv5/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense1/kerneldense1/biasdense/kernel
dense/bias	iterationlearning_rateAdam/m/conv1/kernelAdam/v/conv1/kernelAdam/m/conv1/biasAdam/v/conv1/bias Adam/m/batch_normalization/gamma Adam/v/batch_normalization/gammaAdam/m/batch_normalization/betaAdam/v/batch_normalization/betaAdam/m/conv2/kernelAdam/v/conv2/kernelAdam/m/conv2/biasAdam/v/conv2/bias"Adam/m/batch_normalization_1/gamma"Adam/v/batch_normalization_1/gamma!Adam/m/batch_normalization_1/beta!Adam/v/batch_normalization_1/betaAdam/m/conv3/kernelAdam/v/conv3/kernelAdam/m/conv3/biasAdam/v/conv3/bias"Adam/m/batch_normalization_2/gamma"Adam/v/batch_normalization_2/gamma!Adam/m/batch_normalization_2/beta!Adam/v/batch_normalization_2/betaAdam/m/conv4/kernelAdam/v/conv4/kernelAdam/m/conv4/biasAdam/v/conv4/bias"Adam/m/batch_normalization_3/gamma"Adam/v/batch_normalization_3/gamma!Adam/m/batch_normalization_3/beta!Adam/v/batch_normalization_3/betaAdam/m/conv5/kernelAdam/v/conv5/kernelAdam/m/conv5/biasAdam/v/conv5/bias"Adam/m/batch_normalization_4/gamma"Adam/v/batch_normalization_4/gamma!Adam/m/batch_normalization_4/beta!Adam/v/batch_normalization_4/betaAdam/m/dense1/kernelAdam/v/dense1/kernelAdam/m/dense1/biasAdam/v/dense1/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biastotal_1count_1totalcount*d
Tin]
[2Y*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_46011▀Щ
я
`
'__inference_dropout_layer_call_fn_45389

inputs
identityИвStatefulPartitionedCall╛
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_43700p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_45247

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╚
a
E__inference_sequential_layer_call_and_return_conditional_losses_42948

inputs
identity╜
resize/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_resize_layer_call_and_return_conditional_losses_42935╪
rescale/PartitionedCallPartitionedCallresize/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_rescale_layer_call_and_return_conditional_losses_42945r
IdentityIdentity rescale/PartitionedCall:output:0*
T0*1
_output_shapes
:         ЦЦ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЦЦ:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
╔
Щ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_43018

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ъ
^
B__inference_re_lu_4_layer_call_and_return_conditional_losses_43546

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:         Аc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_43221

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╦
Ы
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43246

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ъ
^
B__inference_re_lu_4_layer_call_and_return_conditional_losses_45338

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:         Аc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
║╞
╫#
 __inference__wrapped_model_42922
sequential_inputK
1sequential_1_conv1_conv2d_readvariableop_resource: @
2sequential_1_conv1_biasadd_readvariableop_resource: F
8sequential_1_batch_normalization_readvariableop_resource: H
:sequential_1_batch_normalization_readvariableop_1_resource: W
Isequential_1_batch_normalization_fusedbatchnormv3_readvariableop_resource: Y
Ksequential_1_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: K
1sequential_1_conv2_conv2d_readvariableop_resource:  @
2sequential_1_conv2_biasadd_readvariableop_resource: H
:sequential_1_batch_normalization_1_readvariableop_resource: J
<sequential_1_batch_normalization_1_readvariableop_1_resource: Y
Ksequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource: [
Msequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: K
1sequential_1_conv3_conv2d_readvariableop_resource: @@
2sequential_1_conv3_biasadd_readvariableop_resource:@H
:sequential_1_batch_normalization_2_readvariableop_resource:@J
<sequential_1_batch_normalization_2_readvariableop_1_resource:@Y
Ksequential_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@[
Msequential_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@K
1sequential_1_conv4_conv2d_readvariableop_resource:@@@
2sequential_1_conv4_biasadd_readvariableop_resource:@H
:sequential_1_batch_normalization_3_readvariableop_resource:@J
<sequential_1_batch_normalization_3_readvariableop_1_resource:@Y
Ksequential_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:@[
Msequential_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:@L
1sequential_1_conv5_conv2d_readvariableop_resource:@АA
2sequential_1_conv5_biasadd_readvariableop_resource:	АI
:sequential_1_batch_normalization_4_readvariableop_resource:	АK
<sequential_1_batch_normalization_4_readvariableop_1_resource:	АZ
Ksequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	А\
Msequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	АF
2sequential_1_dense1_matmul_readvariableop_resource:
ААB
3sequential_1_dense1_biasadd_readvariableop_resource:	АE
1sequential_1_dense_matmul_readvariableop_resource:
А╞
A
2sequential_1_dense_biasadd_readvariableop_resource:	╞

identityИв@sequential_1/batch_normalization/FusedBatchNormV3/ReadVariableOpвBsequential_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1в/sequential_1/batch_normalization/ReadVariableOpв1sequential_1/batch_normalization/ReadVariableOp_1вBsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpвDsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в1sequential_1/batch_normalization_1/ReadVariableOpв3sequential_1/batch_normalization_1/ReadVariableOp_1вBsequential_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpвDsequential_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в1sequential_1/batch_normalization_2/ReadVariableOpв3sequential_1/batch_normalization_2/ReadVariableOp_1вBsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpвDsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в1sequential_1/batch_normalization_3/ReadVariableOpв3sequential_1/batch_normalization_3/ReadVariableOp_1вBsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpвDsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1в1sequential_1/batch_normalization_4/ReadVariableOpв3sequential_1/batch_normalization_4/ReadVariableOp_1в)sequential_1/conv1/BiasAdd/ReadVariableOpв(sequential_1/conv1/Conv2D/ReadVariableOpв)sequential_1/conv2/BiasAdd/ReadVariableOpв(sequential_1/conv2/Conv2D/ReadVariableOpв)sequential_1/conv3/BiasAdd/ReadVariableOpв(sequential_1/conv3/Conv2D/ReadVariableOpв)sequential_1/conv4/BiasAdd/ReadVariableOpв(sequential_1/conv4/Conv2D/ReadVariableOpв)sequential_1/conv5/BiasAdd/ReadVariableOpв(sequential_1/conv5/Conv2D/ReadVariableOpв)sequential_1/dense/BiasAdd/ReadVariableOpв(sequential_1/dense/MatMul/ReadVariableOpв*sequential_1/dense1/BiasAdd/ReadVariableOpв)sequential_1/dense1/MatMul/ReadVariableOp{
*sequential_1/sequential/resize/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Ц   Ц   у
4sequential_1/sequential/resize/resize/ResizeBilinearResizeBilinearsequential_input3sequential_1/sequential/resize/resize/size:output:0*
T0*1
_output_shapes
:         ЦЦ*
half_pixel_centers(k
&sequential_1/sequential/rescale/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;m
(sequential_1/sequential/rescale/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ▐
#sequential_1/sequential/rescale/mulMulEsequential_1/sequential/resize/resize/ResizeBilinear:resized_images:0/sequential_1/sequential/rescale/Cast/x:output:0*
T0*1
_output_shapes
:         ЦЦ─
#sequential_1/sequential/rescale/addAddV2'sequential_1/sequential/rescale/mul:z:01sequential_1/sequential/rescale/Cast_1/x:output:0*
T0*1
_output_shapes
:         ЦЦв
(sequential_1/conv1/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0у
sequential_1/conv1/Conv2DConv2D'sequential_1/sequential/rescale/add:z:00sequential_1/conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ФФ *
paddingVALID*
strides
Ш
)sequential_1/conv1/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╕
sequential_1/conv1/BiasAddBiasAdd"sequential_1/conv1/Conv2D:output:01sequential_1/conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ФФ д
/sequential_1/batch_normalization/ReadVariableOpReadVariableOp8sequential_1_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0и
1sequential_1/batch_normalization/ReadVariableOp_1ReadVariableOp:sequential_1_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0╞
@sequential_1/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_1_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╩
Bsequential_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_1_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0·
1sequential_1/batch_normalization/FusedBatchNormV3FusedBatchNormV3#sequential_1/conv1/BiasAdd:output:07sequential_1/batch_normalization/ReadVariableOp:value:09sequential_1/batch_normalization/ReadVariableOp_1:value:0Hsequential_1/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Jsequential_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ФФ : : : : :*
epsilon%oГ:*
is_training( Т
sequential_1/re_lu/ReluRelu5sequential_1/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         ФФ ┴
"sequential_1/max_pooling2d/MaxPoolMaxPool%sequential_1/re_lu/Relu:activations:0*/
_output_shapes
:         JJ *
ksize
*
paddingVALID*
strides
в
(sequential_1/conv2/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0х
sequential_1/conv2/Conv2DConv2D+sequential_1/max_pooling2d/MaxPool:output:00sequential_1/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         HH *
paddingVALID*
strides
Ш
)sequential_1/conv2/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╢
sequential_1/conv2/BiasAddBiasAdd"sequential_1/conv2/Conv2D:output:01sequential_1/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         HH и
1sequential_1/batch_normalization_1/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype0м
3sequential_1/batch_normalization_1/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype0╩
Bsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╬
Dsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0В
3sequential_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3#sequential_1/conv2/BiasAdd:output:09sequential_1/batch_normalization_1/ReadVariableOp:value:0;sequential_1/batch_normalization_1/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         HH : : : : :*
epsilon%oГ:*
is_training( Ф
sequential_1/re_lu_1/ReluRelu7sequential_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         HH ┼
$sequential_1/max_pooling2d_1/MaxPoolMaxPool'sequential_1/re_lu_1/Relu:activations:0*/
_output_shapes
:         $$ *
ksize
*
paddingVALID*
strides
в
(sequential_1/conv3/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0ч
sequential_1/conv3/Conv2DConv2D-sequential_1/max_pooling2d_1/MaxPool:output:00sequential_1/conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ""@*
paddingVALID*
strides
Ш
)sequential_1/conv3/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╢
sequential_1/conv3/BiasAddBiasAdd"sequential_1/conv3/Conv2D:output:01sequential_1/conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ""@и
1sequential_1/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0м
3sequential_1/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0╩
Bsequential_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0╬
Dsequential_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0В
3sequential_1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3#sequential_1/conv3/BiasAdd:output:09sequential_1/batch_normalization_2/ReadVariableOp:value:0;sequential_1/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         ""@:@:@:@:@:*
epsilon%oГ:*
is_training( Ф
sequential_1/re_lu_2/ReluRelu7sequential_1/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         ""@┼
$sequential_1/max_pooling2d_2/MaxPoolMaxPool'sequential_1/re_lu_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
в
(sequential_1/conv4/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ч
sequential_1/conv4/Conv2DConv2D-sequential_1/max_pooling2d_2/MaxPool:output:00sequential_1/conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
Ш
)sequential_1/conv4/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╢
sequential_1/conv4/BiasAddBiasAdd"sequential_1/conv4/Conv2D:output:01sequential_1/conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @и
1sequential_1/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0м
3sequential_1/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╩
Bsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0╬
Dsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0В
3sequential_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3#sequential_1/conv4/BiasAdd:output:09sequential_1/batch_normalization_3/ReadVariableOp:value:0;sequential_1/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( Ф
sequential_1/re_lu_3/ReluRelu7sequential_1/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @┼
$sequential_1/max_pooling2d_3/MaxPoolMaxPool'sequential_1/re_lu_3/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
г
(sequential_1/conv5/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv5_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0ш
sequential_1/conv5/Conv2DConv2D-sequential_1/max_pooling2d_3/MaxPool:output:00sequential_1/conv5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
Щ
)sequential_1/conv5/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
sequential_1/conv5/BiasAddBiasAdd"sequential_1/conv5/Conv2D:output:01sequential_1/conv5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ай
1sequential_1/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_4_readvariableop_resource*
_output_shapes	
:А*
dtype0н
3sequential_1/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╦
Bsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╧
Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0З
3sequential_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3#sequential_1/conv5/BiasAdd:output:09sequential_1/batch_normalization_4/ReadVariableOp:value:0;sequential_1/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( Х
sequential_1/re_lu_4/ReluRelu7sequential_1/batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         А╞
$sequential_1/max_pooling2d_4/MaxPoolMaxPool'sequential_1/re_lu_4/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
k
sequential_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       о
sequential_1/flatten/ReshapeReshape-sequential_1/max_pooling2d_4/MaxPool:output:0#sequential_1/flatten/Const:output:0*
T0*(
_output_shapes
:         АЮ
)sequential_1/dense1/MatMul/ReadVariableOpReadVariableOp2sequential_1_dense1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0▒
sequential_1/dense1/MatMulMatMul%sequential_1/flatten/Reshape:output:01sequential_1/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЫ
*sequential_1/dense1/BiasAdd/ReadVariableOpReadVariableOp3sequential_1_dense1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0│
sequential_1/dense1/BiasAddBiasAdd$sequential_1/dense1/MatMul:product:02sequential_1/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аy
sequential_1/dense1/ReluRelu$sequential_1/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:         АД
sequential_1/dropout/IdentityIdentity&sequential_1/dense1/Relu:activations:0*
T0*(
_output_shapes
:         АЬ
(sequential_1/dense/MatMul/ReadVariableOpReadVariableOp1sequential_1_dense_matmul_readvariableop_resource* 
_output_shapes
:
А╞
*
dtype0░
sequential_1/dense/MatMulMatMul&sequential_1/dropout/Identity:output:00sequential_1/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╞
Щ
)sequential_1/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_dense_biasadd_readvariableop_resource*
_output_shapes	
:╞
*
dtype0░
sequential_1/dense/BiasAddBiasAdd#sequential_1/dense/MatMul:product:01sequential_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╞
}
sequential_1/dense/SoftmaxSoftmax#sequential_1/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         ╞
t
IdentityIdentity$sequential_1/dense/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:         ╞
я
NoOpNoOpA^sequential_1/batch_normalization/FusedBatchNormV3/ReadVariableOpC^sequential_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_10^sequential_1/batch_normalization/ReadVariableOp2^sequential_1/batch_normalization/ReadVariableOp_1C^sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_1/ReadVariableOp4^sequential_1/batch_normalization_1/ReadVariableOp_1C^sequential_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_2/ReadVariableOp4^sequential_1/batch_normalization_2/ReadVariableOp_1C^sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_3/ReadVariableOp4^sequential_1/batch_normalization_3/ReadVariableOp_1C^sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_4/ReadVariableOp4^sequential_1/batch_normalization_4/ReadVariableOp_1*^sequential_1/conv1/BiasAdd/ReadVariableOp)^sequential_1/conv1/Conv2D/ReadVariableOp*^sequential_1/conv2/BiasAdd/ReadVariableOp)^sequential_1/conv2/Conv2D/ReadVariableOp*^sequential_1/conv3/BiasAdd/ReadVariableOp)^sequential_1/conv3/Conv2D/ReadVariableOp*^sequential_1/conv4/BiasAdd/ReadVariableOp)^sequential_1/conv4/Conv2D/ReadVariableOp*^sequential_1/conv5/BiasAdd/ReadVariableOp)^sequential_1/conv5/Conv2D/ReadVariableOp*^sequential_1/dense/BiasAdd/ReadVariableOp)^sequential_1/dense/MatMul/ReadVariableOp+^sequential_1/dense1/BiasAdd/ReadVariableOp*^sequential_1/dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Д
@sequential_1/batch_normalization/FusedBatchNormV3/ReadVariableOp@sequential_1/batch_normalization/FusedBatchNormV3/ReadVariableOp2И
Bsequential_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Bsequential_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_12b
/sequential_1/batch_normalization/ReadVariableOp/sequential_1/batch_normalization/ReadVariableOp2f
1sequential_1/batch_normalization/ReadVariableOp_11sequential_1/batch_normalization/ReadVariableOp_12И
Bsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2М
Dsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12f
1sequential_1/batch_normalization_1/ReadVariableOp1sequential_1/batch_normalization_1/ReadVariableOp2j
3sequential_1/batch_normalization_1/ReadVariableOp_13sequential_1/batch_normalization_1/ReadVariableOp_12И
Bsequential_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2М
Dsequential_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12f
1sequential_1/batch_normalization_2/ReadVariableOp1sequential_1/batch_normalization_2/ReadVariableOp2j
3sequential_1/batch_normalization_2/ReadVariableOp_13sequential_1/batch_normalization_2/ReadVariableOp_12И
Bsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2М
Dsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12f
1sequential_1/batch_normalization_3/ReadVariableOp1sequential_1/batch_normalization_3/ReadVariableOp2j
3sequential_1/batch_normalization_3/ReadVariableOp_13sequential_1/batch_normalization_3/ReadVariableOp_12И
Bsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2М
Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_1/batch_normalization_4/ReadVariableOp1sequential_1/batch_normalization_4/ReadVariableOp2j
3sequential_1/batch_normalization_4/ReadVariableOp_13sequential_1/batch_normalization_4/ReadVariableOp_12V
)sequential_1/conv1/BiasAdd/ReadVariableOp)sequential_1/conv1/BiasAdd/ReadVariableOp2T
(sequential_1/conv1/Conv2D/ReadVariableOp(sequential_1/conv1/Conv2D/ReadVariableOp2V
)sequential_1/conv2/BiasAdd/ReadVariableOp)sequential_1/conv2/BiasAdd/ReadVariableOp2T
(sequential_1/conv2/Conv2D/ReadVariableOp(sequential_1/conv2/Conv2D/ReadVariableOp2V
)sequential_1/conv3/BiasAdd/ReadVariableOp)sequential_1/conv3/BiasAdd/ReadVariableOp2T
(sequential_1/conv3/Conv2D/ReadVariableOp(sequential_1/conv3/Conv2D/ReadVariableOp2V
)sequential_1/conv4/BiasAdd/ReadVariableOp)sequential_1/conv4/BiasAdd/ReadVariableOp2T
(sequential_1/conv4/Conv2D/ReadVariableOp(sequential_1/conv4/Conv2D/ReadVariableOp2V
)sequential_1/conv5/BiasAdd/ReadVariableOp)sequential_1/conv5/BiasAdd/ReadVariableOp2T
(sequential_1/conv5/Conv2D/ReadVariableOp(sequential_1/conv5/Conv2D/ReadVariableOp2V
)sequential_1/dense/BiasAdd/ReadVariableOp)sequential_1/dense/BiasAdd/ReadVariableOp2T
(sequential_1/dense/MatMul/ReadVariableOp(sequential_1/dense/MatMul/ReadVariableOp2X
*sequential_1/dense1/BiasAdd/ReadVariableOp*sequential_1/dense1/BiasAdd/ReadVariableOp2V
)sequential_1/dense1/MatMul/ReadVariableOp)sequential_1/dense1/MatMul/ReadVariableOp:c _
1
_output_shapes
:         ЦЦ
*
_user_specified_namesequential_input
д

ї
A__inference_dense1_layer_call_and_return_conditional_losses_45379

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┘
`
B__inference_dropout_layer_call_and_return_conditional_losses_45394

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Н	
╨
5__inference_batch_normalization_3_layer_call_fn_45191

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43277Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
┴
Х
%__inference_dense_layer_call_fn_45415

inputs
unknown:
А╞

	unknown_0:	╞

identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╞
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_43592p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╞
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ї
^
B__inference_rescale_layer_call_and_return_conditional_losses_42945

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:         ЦЦd
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:         ЦЦY
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:         ЦЦ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЦЦ:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
ц
^
B__inference_re_lu_1_layer_call_and_return_conditional_losses_45035

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         HH b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         HH "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         HH :W S
/
_output_shapes
:         HH 
 
_user_specified_nameinputs
л
]
A__inference_resize_layer_call_and_return_conditional_losses_42935

inputs
identity\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Ц   Ц   Ы
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*1
_output_shapes
:         ЦЦ*
half_pixel_centers(x
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:         ЦЦ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЦЦ:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
Е
┐
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_45126

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╬
Ь
,__inference_sequential_1_layer_call_fn_44532

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@%

unknown_23:@А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:
АА

unknown_30:	А

unknown_31:
А╞


unknown_32:	╞

identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╞
*:
_read_only_resource_inputs
	
 !"*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_43971p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╞
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
┐
B
&__inference_resize_layer_call_fn_45431

inputs
identity╢
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_resize_layer_call_and_return_conditional_losses_42935j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ЦЦ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЦЦ:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
┌l
ў
G__inference_sequential_1_layer_call_and_return_conditional_losses_43971

inputs%
conv1_43878: 
conv1_43880: '
batch_normalization_43883: '
batch_normalization_43885: '
batch_normalization_43887: '
batch_normalization_43889: %
conv2_43894:  
conv2_43896: )
batch_normalization_1_43899: )
batch_normalization_1_43901: )
batch_normalization_1_43903: )
batch_normalization_1_43905: %
conv3_43910: @
conv3_43912:@)
batch_normalization_2_43915:@)
batch_normalization_2_43917:@)
batch_normalization_2_43919:@)
batch_normalization_2_43921:@%
conv4_43926:@@
conv4_43928:@)
batch_normalization_3_43931:@)
batch_normalization_3_43933:@)
batch_normalization_3_43935:@)
batch_normalization_3_43937:@&
conv5_43942:@А
conv5_43944:	А*
batch_normalization_4_43947:	А*
batch_normalization_4_43949:	А*
batch_normalization_4_43951:	А*
batch_normalization_4_43953:	А 
dense1_43959:
АА
dense1_43961:	А
dense_43965:
А╞

dense_43967:	╞

identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallвconv1/StatefulPartitionedCallвconv2/StatefulPartitionedCallвconv3/StatefulPartitionedCallвconv4/StatefulPartitionedCallвconv5/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense1/StatefulPartitionedCallвdropout/StatefulPartitionedCall┼
sequential/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_42976И
conv1/StatefulPartitionedCallStatefulPartitionedCall#sequential/PartitionedCall:output:0conv1_43878conv1_43880*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ФФ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_43394√
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0batch_normalization_43883batch_normalization_43885batch_normalization_43887batch_normalization_43889*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ФФ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_43049щ
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ФФ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_43414с
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         JJ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_43069Й
conv2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2_43894conv2_43896*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         HH *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_43427Е
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0batch_normalization_1_43899batch_normalization_1_43901batch_normalization_1_43903batch_normalization_1_43905*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         HH *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43125э
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         HH * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_43447ч
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         $$ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_43145Л
conv3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv3_43910conv3_43912*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ""@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv3_layer_call_and_return_conditional_losses_43460Е
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0batch_normalization_2_43915batch_normalization_2_43917batch_normalization_2_43919batch_normalization_2_43921*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ""@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43201э
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ""@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_43480ч
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_43221Л
conv4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv4_43926conv4_43928*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv4_layer_call_and_return_conditional_losses_43493Е
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall&conv4/StatefulPartitionedCall:output:0batch_normalization_3_43931batch_normalization_3_43933batch_normalization_3_43935batch_normalization_3_43937*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43277э
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_43513ч
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_43297М
conv5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv5_43942conv5_43944*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv5_layer_call_and_return_conditional_losses_43526Ж
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&conv5/StatefulPartitionedCall:output:0batch_normalization_4_43947batch_normalization_4_43949batch_normalization_4_43951batch_normalization_4_43953*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43353ю
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_43546ш
max_pooling2d_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_43373╪
flatten/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_43555А
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_43959dense1_43961*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_43568ч
dropout/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_43700Д
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_43965dense_43967*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╞
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_43592v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╞
╖
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^conv5/StatefulPartitionedCall^dense/StatefulPartitionedCall^dense1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2>
conv5/StatefulPartitionedCallconv5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
ц
^
B__inference_re_lu_3_layer_call_and_return_conditional_losses_45237

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         @b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Э
C
'__inference_dropout_layer_call_fn_45384

inputs
identityо
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_43579a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ь
ж
,__inference_sequential_1_layer_call_fn_44115
sequential_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@%

unknown_23:@А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:
АА

unknown_30:	А

unknown_31:
А╞


unknown_32:	╞

identityИвStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╞
*:
_read_only_resource_inputs
	
 !"*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_43971p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╞
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:         ЦЦ
*
_user_specified_namesequential_input
П	
╨
5__inference_batch_normalization_1_layer_call_fn_44976

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43094Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
П	
╨
5__inference_batch_normalization_2_layer_call_fn_45077

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43170Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╜
A
%__inference_re_lu_layer_call_fn_44929

inputs
identity╡
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ФФ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_43414j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ФФ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ФФ :Y U
1
_output_shapes
:         ФФ 
 
_user_specified_nameinputs
Х
├
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_45328

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╪Є
п9
!__inference__traced_restore_46011
file_prefix7
assignvariableop_conv1_kernel: +
assignvariableop_1_conv1_bias: :
,assignvariableop_2_batch_normalization_gamma: 9
+assignvariableop_3_batch_normalization_beta: @
2assignvariableop_4_batch_normalization_moving_mean: D
6assignvariableop_5_batch_normalization_moving_variance: 9
assignvariableop_6_conv2_kernel:  +
assignvariableop_7_conv2_bias: <
.assignvariableop_8_batch_normalization_1_gamma: ;
-assignvariableop_9_batch_normalization_1_beta: C
5assignvariableop_10_batch_normalization_1_moving_mean: G
9assignvariableop_11_batch_normalization_1_moving_variance: :
 assignvariableop_12_conv3_kernel: @,
assignvariableop_13_conv3_bias:@=
/assignvariableop_14_batch_normalization_2_gamma:@<
.assignvariableop_15_batch_normalization_2_beta:@C
5assignvariableop_16_batch_normalization_2_moving_mean:@G
9assignvariableop_17_batch_normalization_2_moving_variance:@:
 assignvariableop_18_conv4_kernel:@@,
assignvariableop_19_conv4_bias:@=
/assignvariableop_20_batch_normalization_3_gamma:@<
.assignvariableop_21_batch_normalization_3_beta:@C
5assignvariableop_22_batch_normalization_3_moving_mean:@G
9assignvariableop_23_batch_normalization_3_moving_variance:@;
 assignvariableop_24_conv5_kernel:@А-
assignvariableop_25_conv5_bias:	А>
/assignvariableop_26_batch_normalization_4_gamma:	А=
.assignvariableop_27_batch_normalization_4_beta:	АD
5assignvariableop_28_batch_normalization_4_moving_mean:	АH
9assignvariableop_29_batch_normalization_4_moving_variance:	А5
!assignvariableop_30_dense1_kernel:
АА.
assignvariableop_31_dense1_bias:	А4
 assignvariableop_32_dense_kernel:
А╞
-
assignvariableop_33_dense_bias:	╞
'
assignvariableop_34_iteration:	 +
!assignvariableop_35_learning_rate: A
'assignvariableop_36_adam_m_conv1_kernel: A
'assignvariableop_37_adam_v_conv1_kernel: 3
%assignvariableop_38_adam_m_conv1_bias: 3
%assignvariableop_39_adam_v_conv1_bias: B
4assignvariableop_40_adam_m_batch_normalization_gamma: B
4assignvariableop_41_adam_v_batch_normalization_gamma: A
3assignvariableop_42_adam_m_batch_normalization_beta: A
3assignvariableop_43_adam_v_batch_normalization_beta: A
'assignvariableop_44_adam_m_conv2_kernel:  A
'assignvariableop_45_adam_v_conv2_kernel:  3
%assignvariableop_46_adam_m_conv2_bias: 3
%assignvariableop_47_adam_v_conv2_bias: D
6assignvariableop_48_adam_m_batch_normalization_1_gamma: D
6assignvariableop_49_adam_v_batch_normalization_1_gamma: C
5assignvariableop_50_adam_m_batch_normalization_1_beta: C
5assignvariableop_51_adam_v_batch_normalization_1_beta: A
'assignvariableop_52_adam_m_conv3_kernel: @A
'assignvariableop_53_adam_v_conv3_kernel: @3
%assignvariableop_54_adam_m_conv3_bias:@3
%assignvariableop_55_adam_v_conv3_bias:@D
6assignvariableop_56_adam_m_batch_normalization_2_gamma:@D
6assignvariableop_57_adam_v_batch_normalization_2_gamma:@C
5assignvariableop_58_adam_m_batch_normalization_2_beta:@C
5assignvariableop_59_adam_v_batch_normalization_2_beta:@A
'assignvariableop_60_adam_m_conv4_kernel:@@A
'assignvariableop_61_adam_v_conv4_kernel:@@3
%assignvariableop_62_adam_m_conv4_bias:@3
%assignvariableop_63_adam_v_conv4_bias:@D
6assignvariableop_64_adam_m_batch_normalization_3_gamma:@D
6assignvariableop_65_adam_v_batch_normalization_3_gamma:@C
5assignvariableop_66_adam_m_batch_normalization_3_beta:@C
5assignvariableop_67_adam_v_batch_normalization_3_beta:@B
'assignvariableop_68_adam_m_conv5_kernel:@АB
'assignvariableop_69_adam_v_conv5_kernel:@А4
%assignvariableop_70_adam_m_conv5_bias:	А4
%assignvariableop_71_adam_v_conv5_bias:	АE
6assignvariableop_72_adam_m_batch_normalization_4_gamma:	АE
6assignvariableop_73_adam_v_batch_normalization_4_gamma:	АD
5assignvariableop_74_adam_m_batch_normalization_4_beta:	АD
5assignvariableop_75_adam_v_batch_normalization_4_beta:	А<
(assignvariableop_76_adam_m_dense1_kernel:
АА<
(assignvariableop_77_adam_v_dense1_kernel:
АА5
&assignvariableop_78_adam_m_dense1_bias:	А5
&assignvariableop_79_adam_v_dense1_bias:	А;
'assignvariableop_80_adam_m_dense_kernel:
А╞
;
'assignvariableop_81_adam_v_dense_kernel:
А╞
4
%assignvariableop_82_adam_m_dense_bias:	╞
4
%assignvariableop_83_adam_v_dense_bias:	╞
%
assignvariableop_84_total_1: %
assignvariableop_85_count_1: #
assignvariableop_86_total: #
assignvariableop_87_count: 
identity_89ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_75вAssignVariableOp_76вAssignVariableOp_77вAssignVariableOp_78вAssignVariableOp_79вAssignVariableOp_8вAssignVariableOp_80вAssignVariableOp_81вAssignVariableOp_82вAssignVariableOp_83вAssignVariableOp_84вAssignVariableOp_85вAssignVariableOp_86вAssignVariableOp_87вAssignVariableOp_9а&
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*╞%
value╝%B╣%YB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHе
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*╟
value╜B║YB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ▐
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*·
_output_shapesч
ф:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*g
dtypes]
[2Y	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:░
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╔
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv2_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv2_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_12AssignVariableOp assignvariableop_12_conv3_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_13AssignVariableOpassignvariableop_13_conv3_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_18AssignVariableOp assignvariableop_18_conv4_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_19AssignVariableOpassignvariableop_19_conv4_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_24AssignVariableOp assignvariableop_24_conv5_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_25AssignVariableOpassignvariableop_25_conv5_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_4_gammaIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_4_betaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_4_moving_meanIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_4_moving_varianceIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_30AssignVariableOp!assignvariableop_30_dense1_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_31AssignVariableOpassignvariableop_31_dense1_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_32AssignVariableOp assignvariableop_32_dense_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_33AssignVariableOpassignvariableop_33_dense_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0	*
_output_shapes
:╢
AssignVariableOp_34AssignVariableOpassignvariableop_34_iterationIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_35AssignVariableOp!assignvariableop_35_learning_rateIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_m_conv1_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_v_conv1_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_38AssignVariableOp%assignvariableop_38_adam_m_conv1_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_39AssignVariableOp%assignvariableop_39_adam_v_conv1_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_40AssignVariableOp4assignvariableop_40_adam_m_batch_normalization_gammaIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_41AssignVariableOp4assignvariableop_41_adam_v_batch_normalization_gammaIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_42AssignVariableOp3assignvariableop_42_adam_m_batch_normalization_betaIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_43AssignVariableOp3assignvariableop_43_adam_v_batch_normalization_betaIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_m_conv2_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_v_conv2_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_m_conv2_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_47AssignVariableOp%assignvariableop_47_adam_v_conv2_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_48AssignVariableOp6assignvariableop_48_adam_m_batch_normalization_1_gammaIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_v_batch_normalization_1_gammaIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_m_batch_normalization_1_betaIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_51AssignVariableOp5assignvariableop_51_adam_v_batch_normalization_1_betaIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_m_conv3_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_53AssignVariableOp'assignvariableop_53_adam_v_conv3_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_54AssignVariableOp%assignvariableop_54_adam_m_conv3_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_55AssignVariableOp%assignvariableop_55_adam_v_conv3_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_56AssignVariableOp6assignvariableop_56_adam_m_batch_normalization_2_gammaIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_v_batch_normalization_2_gammaIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_m_batch_normalization_2_betaIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_59AssignVariableOp5assignvariableop_59_adam_v_batch_normalization_2_betaIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_m_conv4_kernelIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_61AssignVariableOp'assignvariableop_61_adam_v_conv4_kernelIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_62AssignVariableOp%assignvariableop_62_adam_m_conv4_biasIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_63AssignVariableOp%assignvariableop_63_adam_v_conv4_biasIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_m_batch_normalization_3_gammaIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_v_batch_normalization_3_gammaIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_m_batch_normalization_3_betaIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_67AssignVariableOp5assignvariableop_67_adam_v_batch_normalization_3_betaIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_68AssignVariableOp'assignvariableop_68_adam_m_conv5_kernelIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_69AssignVariableOp'assignvariableop_69_adam_v_conv5_kernelIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_70AssignVariableOp%assignvariableop_70_adam_m_conv5_biasIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_71AssignVariableOp%assignvariableop_71_adam_v_conv5_biasIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_m_batch_normalization_4_gammaIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_73AssignVariableOp6assignvariableop_73_adam_v_batch_normalization_4_gammaIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_74AssignVariableOp5assignvariableop_74_adam_m_batch_normalization_4_betaIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_75AssignVariableOp5assignvariableop_75_adam_v_batch_normalization_4_betaIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_m_dense1_kernelIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_77AssignVariableOp(assignvariableop_77_adam_v_dense1_kernelIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_78AssignVariableOp&assignvariableop_78_adam_m_dense1_biasIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_79AssignVariableOp&assignvariableop_79_adam_v_dense1_biasIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_80AssignVariableOp'assignvariableop_80_adam_m_dense_kernelIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_81AssignVariableOp'assignvariableop_81_adam_v_dense_kernelIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_82AssignVariableOp%assignvariableop_82_adam_m_dense_biasIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_83AssignVariableOp%assignvariableop_83_adam_v_dense_biasIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_84AssignVariableOpassignvariableop_84_total_1Identity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_85AssignVariableOpassignvariableop_85_count_1Identity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_86AssignVariableOpassignvariableop_86_totalIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_87AssignVariableOpassignvariableop_87_countIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ▀
Identity_88Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_89IdentityIdentity_88:output:0^NoOp_1*
T0*
_output_shapes
: ╠
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_89Identity_89:output:0*╟
_input_shapes╡
▓: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Т
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_43145

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_45348

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╦
Ы
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_45007

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
д

∙
@__inference_conv3_layer_call_and_return_conditional_losses_43460

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ""@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ""@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         ""@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         $$ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         $$ 
 
_user_specified_nameinputs
П

a
B__inference_dropout_layer_call_and_return_conditional_losses_43700

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╣
C
'__inference_re_lu_1_layer_call_fn_45030

inputs
identity╡
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         HH * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_43447h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         HH "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         HH :W S
/
_output_shapes
:         HH 
 
_user_specified_nameinputs
Р
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_44944

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
и

Ї
@__inference_dense_layer_call_and_return_conditional_losses_45426

inputs2
matmul_readvariableop_resource:
А╞
.
biasadd_readvariableop_resource:	╞

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А╞
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╞
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╞
*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╞
W
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:         ╞
a
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:         ╞
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
─k
╒
G__inference_sequential_1_layer_call_and_return_conditional_losses_43599

inputs%
conv1_43395: 
conv1_43397: '
batch_normalization_43400: '
batch_normalization_43402: '
batch_normalization_43404: '
batch_normalization_43406: %
conv2_43428:  
conv2_43430: )
batch_normalization_1_43433: )
batch_normalization_1_43435: )
batch_normalization_1_43437: )
batch_normalization_1_43439: %
conv3_43461: @
conv3_43463:@)
batch_normalization_2_43466:@)
batch_normalization_2_43468:@)
batch_normalization_2_43470:@)
batch_normalization_2_43472:@%
conv4_43494:@@
conv4_43496:@)
batch_normalization_3_43499:@)
batch_normalization_3_43501:@)
batch_normalization_3_43503:@)
batch_normalization_3_43505:@&
conv5_43527:@А
conv5_43529:	А*
batch_normalization_4_43532:	А*
batch_normalization_4_43534:	А*
batch_normalization_4_43536:	А*
batch_normalization_4_43538:	А 
dense1_43569:
АА
dense1_43571:	А
dense_43593:
А╞

dense_43595:	╞

identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallвconv1/StatefulPartitionedCallвconv2/StatefulPartitionedCallвconv3/StatefulPartitionedCallвconv4/StatefulPartitionedCallвconv5/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense1/StatefulPartitionedCall┼
sequential/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_42948И
conv1/StatefulPartitionedCallStatefulPartitionedCall#sequential/PartitionedCall:output:0conv1_43395conv1_43397*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ФФ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_43394¤
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0batch_normalization_43400batch_normalization_43402batch_normalization_43404batch_normalization_43406*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ФФ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_43018щ
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ФФ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_43414с
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         JJ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_43069Й
conv2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2_43428conv2_43430*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         HH *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_43427З
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0batch_normalization_1_43433batch_normalization_1_43435batch_normalization_1_43437batch_normalization_1_43439*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         HH *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43094э
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         HH * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_43447ч
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         $$ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_43145Л
conv3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv3_43461conv3_43463*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ""@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv3_layer_call_and_return_conditional_losses_43460З
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0batch_normalization_2_43466batch_normalization_2_43468batch_normalization_2_43470batch_normalization_2_43472*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ""@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43170э
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ""@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_43480ч
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_43221Л
conv4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv4_43494conv4_43496*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv4_layer_call_and_return_conditional_losses_43493З
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall&conv4/StatefulPartitionedCall:output:0batch_normalization_3_43499batch_normalization_3_43501batch_normalization_3_43503batch_normalization_3_43505*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43246э
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_43513ч
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_43297М
conv5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv5_43527conv5_43529*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv5_layer_call_and_return_conditional_losses_43526И
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&conv5/StatefulPartitionedCall:output:0batch_normalization_4_43532batch_normalization_4_43534batch_normalization_4_43536batch_normalization_4_43538*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43322ю
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_43546ш
max_pooling2d_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_43373╪
flatten/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_43555А
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_43569dense1_43571*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_43568╫
dropout/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_43579№
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_43593dense_43595*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╞
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_43592v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╞
Х
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^conv5/StatefulPartitionedCall^dense/StatefulPartitionedCall^dense1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2>
conv5/StatefulPartitionedCallconv5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
▓
I
-__inference_max_pooling2d_layer_call_fn_44939

inputs
identity╓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_43069Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Л	
╬
3__inference_batch_normalization_layer_call_fn_44875

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_43018Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_45045

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
°l
Б
G__inference_sequential_1_layer_call_and_return_conditional_losses_44309
sequential_input%
conv1_44216: 
conv1_44218: '
batch_normalization_44221: '
batch_normalization_44223: '
batch_normalization_44225: '
batch_normalization_44227: %
conv2_44232:  
conv2_44234: )
batch_normalization_1_44237: )
batch_normalization_1_44239: )
batch_normalization_1_44241: )
batch_normalization_1_44243: %
conv3_44248: @
conv3_44250:@)
batch_normalization_2_44253:@)
batch_normalization_2_44255:@)
batch_normalization_2_44257:@)
batch_normalization_2_44259:@%
conv4_44264:@@
conv4_44266:@)
batch_normalization_3_44269:@)
batch_normalization_3_44271:@)
batch_normalization_3_44273:@)
batch_normalization_3_44275:@&
conv5_44280:@А
conv5_44282:	А*
batch_normalization_4_44285:	А*
batch_normalization_4_44287:	А*
batch_normalization_4_44289:	А*
batch_normalization_4_44291:	А 
dense1_44297:
АА
dense1_44299:	А
dense_44303:
А╞

dense_44305:	╞

identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallвconv1/StatefulPartitionedCallвconv2/StatefulPartitionedCallвconv3/StatefulPartitionedCallвconv4/StatefulPartitionedCallвconv5/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense1/StatefulPartitionedCallвdropout/StatefulPartitionedCall╧
sequential/PartitionedCallPartitionedCallsequential_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_42976И
conv1/StatefulPartitionedCallStatefulPartitionedCall#sequential/PartitionedCall:output:0conv1_44216conv1_44218*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ФФ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_43394√
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0batch_normalization_44221batch_normalization_44223batch_normalization_44225batch_normalization_44227*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ФФ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_43049щ
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ФФ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_43414с
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         JJ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_43069Й
conv2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2_44232conv2_44234*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         HH *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_43427Е
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0batch_normalization_1_44237batch_normalization_1_44239batch_normalization_1_44241batch_normalization_1_44243*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         HH *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43125э
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         HH * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_43447ч
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         $$ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_43145Л
conv3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv3_44248conv3_44250*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ""@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv3_layer_call_and_return_conditional_losses_43460Е
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0batch_normalization_2_44253batch_normalization_2_44255batch_normalization_2_44257batch_normalization_2_44259*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ""@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43201э
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ""@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_43480ч
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_43221Л
conv4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv4_44264conv4_44266*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv4_layer_call_and_return_conditional_losses_43493Е
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall&conv4/StatefulPartitionedCall:output:0batch_normalization_3_44269batch_normalization_3_44271batch_normalization_3_44273batch_normalization_3_44275*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43277э
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_43513ч
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_43297М
conv5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv5_44280conv5_44282*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv5_layer_call_and_return_conditional_losses_43526Ж
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&conv5/StatefulPartitionedCall:output:0batch_normalization_4_44285batch_normalization_4_44287batch_normalization_4_44289batch_normalization_4_44291*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43353ю
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_43546ш
max_pooling2d_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_43373╪
flatten/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_43555А
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_44297dense1_44299*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_43568ч
dropout/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_43700Д
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_44303dense_44305*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╞
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_43592v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╞
╖
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^conv5/StatefulPartitionedCall^dense/StatefulPartitionedCall^dense1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2>
conv5/StatefulPartitionedCallconv5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:c _
1
_output_shapes
:         ЦЦ
*
_user_specified_namesequential_input
ц
^
B__inference_re_lu_3_layer_call_and_return_conditional_losses_43513

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         @b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
о

∙
@__inference_conv1_layer_call_and_return_conditional_losses_44862

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ь
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ФФ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ФФ i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         ФФ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЦЦ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
Р
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_43069

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ў
ж
,__inference_sequential_1_layer_call_fn_43670
sequential_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@%

unknown_23:@А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:
АА

unknown_30:	А

unknown_31:
А╞


unknown_32:	╞

identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╞
*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_43599p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╞
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:         ЦЦ
*
_user_specified_namesequential_input
ц
^
B__inference_re_lu_2_layer_call_and_return_conditional_losses_45136

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         ""@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         ""@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         ""@:W S
/
_output_shapes
:         ""@
 
_user_specified_nameinputs
т
Ъ
%__inference_conv3_layer_call_fn_45054

inputs!
unknown: @
	unknown_0:@
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ""@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv3_layer_call_and_return_conditional_losses_43460w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         ""@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         $$ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         $$ 
 
_user_specified_nameinputs
Е
┐
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43125

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_45146

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
л

√
@__inference_conv5_layer_call_and_return_conditional_losses_43526

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Ї
^
B__inference_rescale_layer_call_and_return_conditional_losses_45450

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:         ЦЦd
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:         ЦЦY
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:         ЦЦ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЦЦ:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
д

∙
@__inference_conv3_layer_call_and_return_conditional_losses_45064

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ""@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ""@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         ""@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         $$ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         $$ 
 
_user_specified_nameinputs
┘
`
B__inference_dropout_layer_call_and_return_conditional_losses_43579

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
═в
А
G__inference_sequential_1_layer_call_and_return_conditional_losses_44669

inputs>
$conv1_conv2d_readvariableop_resource: 3
%conv1_biasadd_readvariableop_resource: 9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: >
$conv2_conv2d_readvariableop_resource:  3
%conv2_biasadd_readvariableop_resource: ;
-batch_normalization_1_readvariableop_resource: =
/batch_normalization_1_readvariableop_1_resource: L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: >
$conv3_conv2d_readvariableop_resource: @3
%conv3_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@>
$conv4_conv2d_readvariableop_resource:@@3
%conv4_biasadd_readvariableop_resource:@;
-batch_normalization_3_readvariableop_resource:@=
/batch_normalization_3_readvariableop_1_resource:@L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:@?
$conv5_conv2d_readvariableop_resource:@А4
%conv5_biasadd_readvariableop_resource:	А<
-batch_normalization_4_readvariableop_resource:	А>
/batch_normalization_4_readvariableop_1_resource:	АM
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	А9
%dense1_matmul_readvariableop_resource:
АА5
&dense1_biasadd_readvariableop_resource:	А8
$dense_matmul_readvariableop_resource:
А╞
4
%dense_biasadd_readvariableop_resource:	╞

identityИв3batch_normalization/FusedBatchNormV3/ReadVariableOpв5batch_normalization/FusedBatchNormV3/ReadVariableOp_1в"batch_normalization/ReadVariableOpв$batch_normalization/ReadVariableOp_1в5batch_normalization_1/FusedBatchNormV3/ReadVariableOpв7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_1/ReadVariableOpв&batch_normalization_1/ReadVariableOp_1в5batch_normalization_2/FusedBatchNormV3/ReadVariableOpв7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_2/ReadVariableOpв&batch_normalization_2/ReadVariableOp_1в5batch_normalization_3/FusedBatchNormV3/ReadVariableOpв7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_3/ReadVariableOpв&batch_normalization_3/ReadVariableOp_1в5batch_normalization_4/FusedBatchNormV3/ReadVariableOpв7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_4/ReadVariableOpв&batch_normalization_4/ReadVariableOp_1вconv1/BiasAdd/ReadVariableOpвconv1/Conv2D/ReadVariableOpвconv2/BiasAdd/ReadVariableOpвconv2/Conv2D/ReadVariableOpвconv3/BiasAdd/ReadVariableOpвconv3/Conv2D/ReadVariableOpвconv4/BiasAdd/ReadVariableOpвconv4/Conv2D/ReadVariableOpвconv5/BiasAdd/ReadVariableOpвconv5/Conv2D/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense1/BiasAdd/ReadVariableOpвdense1/MatMul/ReadVariableOpn
sequential/resize/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Ц   Ц   ┐
'sequential/resize/resize/ResizeBilinearResizeBilinearinputs&sequential/resize/resize/size:output:0*
T0*1
_output_shapes
:         ЦЦ*
half_pixel_centers(^
sequential/rescale/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;`
sequential/rescale/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ╖
sequential/rescale/mulMul8sequential/resize/resize/ResizeBilinear:resized_images:0"sequential/rescale/Cast/x:output:0*
T0*1
_output_shapes
:         ЦЦЭ
sequential/rescale/addAddV2sequential/rescale/mul:z:0$sequential/rescale/Cast_1/x:output:0*
T0*1
_output_shapes
:         ЦЦИ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╝
conv1/Conv2DConv2Dsequential/rescale/add:z:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ФФ *
paddingVALID*
strides
~
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0С
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ФФ К
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0О
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0м
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0░
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0м
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv1/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ФФ : : : : :*
epsilon%oГ:*
is_training( x

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         ФФ з
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*/
_output_shapes
:         JJ *
ksize
*
paddingVALID*
strides
И
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0╛
conv2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         HH *
paddingVALID*
strides
~
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0П
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         HH О
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype0Т
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0┤
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         HH : : : : :*
epsilon%oГ:*
is_training( z
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         HH л
max_pooling2d_1/MaxPoolMaxPoolre_lu_1/Relu:activations:0*/
_output_shapes
:         $$ *
ksize
*
paddingVALID*
strides
И
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0└
conv3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ""@*
paddingVALID*
strides
~
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ""@О
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0Т
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0┤
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv3/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         ""@:@:@:@:@:*
epsilon%oГ:*
is_training( z
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         ""@л
max_pooling2d_2/MaxPoolMaxPoolre_lu_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
И
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0└
conv4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
~
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @О
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0Т
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0┤
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv4/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( z
re_lu_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @л
max_pooling2d_3/MaxPoolMaxPoolre_lu_3/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
Й
conv5/Conv2D/ReadVariableOpReadVariableOp$conv5_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0┴
conv5/Conv2DConv2D max_pooling2d_3/MaxPool:output:0#conv5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides

conv5/BiasAdd/ReadVariableOpReadVariableOp%conv5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Р
conv5/BiasAddBiasAddconv5/Conv2D:output:0$conv5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АП
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:А*
dtype0У
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:А*
dtype0▒
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╡
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╣
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv5/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( {
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         Ам
max_pooling2d_4/MaxPoolMaxPoolre_lu_4/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       З
flatten/ReshapeReshape max_pooling2d_4/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         АД
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0К
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АБ
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0М
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А_
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:         Аj
dropout/IdentityIdentitydense1/Relu:activations:0*
T0*(
_output_shapes
:         АВ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
А╞
*
dtype0Й
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╞

dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:╞
*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╞
c
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*(
_output_shapes
:         ╞
g
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:         ╞
╡
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp^conv5/BiasAdd/ReadVariableOp^conv5/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2<
conv5/BiasAdd/ReadVariableOpconv5/BiasAdd/ReadVariableOp2:
conv5/Conv2D/ReadVariableOpconv5/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
Н	
╨
5__inference_batch_normalization_2_layer_call_fn_45090

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43201Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
о

∙
@__inference_conv1_layer_call_and_return_conditional_losses_43394

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ь
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ФФ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ФФ i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         ФФ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЦЦ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
╟
F
*__inference_sequential_layer_call_fn_44823

inputs
identity║
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_42976j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ЦЦ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЦЦ:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
ц
Ь
%__inference_conv5_layer_call_fn_45256

inputs"
unknown:@А
	unknown_0:	А
identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv5_layer_call_and_return_conditional_losses_43526x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
П

a
B__inference_dropout_layer_call_and_return_conditional_losses_45406

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╢
K
/__inference_max_pooling2d_4_layer_call_fn_45343

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_43373Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ь
\
@__inference_re_lu_layer_call_and_return_conditional_losses_44934

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:         ФФ d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:         ФФ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ФФ :Y U
1
_output_shapes
:         ФФ 
 
_user_specified_nameinputs
Е
┐
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43277

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Н	
╨
5__inference_batch_normalization_1_layer_call_fn_44989

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43125Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
┴
C
'__inference_rescale_layer_call_fn_45442

inputs
identity╖
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_rescale_layer_call_and_return_conditional_losses_42945j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ЦЦ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЦЦ:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
╔
Щ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_44906

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
т
Ъ
%__inference_conv4_layer_call_fn_45155

inputs!
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv4_layer_call_and_return_conditional_losses_43493w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╢
K
/__inference_max_pooling2d_1_layer_call_fn_45040

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_43145Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╪
Ь
,__inference_sequential_1_layer_call_fn_44459

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@%

unknown_23:@А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:
АА

unknown_30:	А

unknown_31:
А╞


unknown_32:	╞

identityИвStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╞
*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_43599p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╞
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
Г
╜
N__inference_batch_normalization_layer_call_and_return_conditional_losses_43049

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
д

∙
@__inference_conv2_layer_call_and_return_conditional_losses_44963

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         HH *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         HH g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         HH w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         JJ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         JJ 
 
_user_specified_nameinputs
╦
Ы
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43094

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ь
\
@__inference_re_lu_layer_call_and_return_conditional_losses_43414

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:         ФФ d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:         ФФ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ФФ :Y U
1
_output_shapes
:         ФФ 
 
_user_specified_nameinputs
╟
F
*__inference_sequential_layer_call_fn_44818

inputs
identity║
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_42948j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ЦЦ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЦЦ:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
╦
Ы
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_45108

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╣
C
'__inference_re_lu_3_layer_call_fn_45232

inputs
identity╡
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_43513h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
тk
▀
G__inference_sequential_1_layer_call_and_return_conditional_losses_44212
sequential_input%
conv1_44119: 
conv1_44121: '
batch_normalization_44124: '
batch_normalization_44126: '
batch_normalization_44128: '
batch_normalization_44130: %
conv2_44135:  
conv2_44137: )
batch_normalization_1_44140: )
batch_normalization_1_44142: )
batch_normalization_1_44144: )
batch_normalization_1_44146: %
conv3_44151: @
conv3_44153:@)
batch_normalization_2_44156:@)
batch_normalization_2_44158:@)
batch_normalization_2_44160:@)
batch_normalization_2_44162:@%
conv4_44167:@@
conv4_44169:@)
batch_normalization_3_44172:@)
batch_normalization_3_44174:@)
batch_normalization_3_44176:@)
batch_normalization_3_44178:@&
conv5_44183:@А
conv5_44185:	А*
batch_normalization_4_44188:	А*
batch_normalization_4_44190:	А*
batch_normalization_4_44192:	А*
batch_normalization_4_44194:	А 
dense1_44200:
АА
dense1_44202:	А
dense_44206:
А╞

dense_44208:	╞

identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallвconv1/StatefulPartitionedCallвconv2/StatefulPartitionedCallвconv3/StatefulPartitionedCallвconv4/StatefulPartitionedCallвconv5/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense1/StatefulPartitionedCall╧
sequential/PartitionedCallPartitionedCallsequential_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_42948И
conv1/StatefulPartitionedCallStatefulPartitionedCall#sequential/PartitionedCall:output:0conv1_44119conv1_44121*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ФФ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_43394¤
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0batch_normalization_44124batch_normalization_44126batch_normalization_44128batch_normalization_44130*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ФФ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_43018щ
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ФФ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_43414с
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         JJ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_43069Й
conv2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2_44135conv2_44137*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         HH *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_43427З
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0batch_normalization_1_44140batch_normalization_1_44142batch_normalization_1_44144batch_normalization_1_44146*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         HH *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43094э
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         HH * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_43447ч
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         $$ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_43145Л
conv3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv3_44151conv3_44153*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ""@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv3_layer_call_and_return_conditional_losses_43460З
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0batch_normalization_2_44156batch_normalization_2_44158batch_normalization_2_44160batch_normalization_2_44162*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ""@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43170э
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ""@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_43480ч
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_43221Л
conv4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv4_44167conv4_44169*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv4_layer_call_and_return_conditional_losses_43493З
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall&conv4/StatefulPartitionedCall:output:0batch_normalization_3_44172batch_normalization_3_44174batch_normalization_3_44176batch_normalization_3_44178*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43246э
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_43513ч
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_43297М
conv5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv5_44183conv5_44185*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv5_layer_call_and_return_conditional_losses_43526И
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&conv5/StatefulPartitionedCall:output:0batch_normalization_4_44188batch_normalization_4_44190batch_normalization_4_44192batch_normalization_4_44194*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43322ю
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_43546ш
max_pooling2d_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_43373╪
flatten/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_43555А
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_44200dense1_44202*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_43568╫
dropout/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_43579№
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_44206dense_44208*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╞
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_43592v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╞
Х
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^conv5/StatefulPartitionedCall^dense/StatefulPartitionedCall^dense1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2>
conv5/StatefulPartitionedCallconv5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall:c _
1
_output_shapes
:         ЦЦ
*
_user_specified_namesequential_input
╦
Ы
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43170

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ц
^
B__inference_re_lu_2_layer_call_and_return_conditional_losses_43480

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         ""@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         ""@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         ""@:W S
/
_output_shapes
:         ""@
 
_user_specified_nameinputs
Г
╜
N__inference_batch_normalization_layer_call_and_return_conditional_losses_44924

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
┌
g
E__inference_sequential_layer_call_and_return_conditional_losses_42996
resize_input
identity├
resize/PartitionedCallPartitionedCallresize_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_resize_layer_call_and_return_conditional_losses_42935╪
rescale/PartitionedCallPartitionedCallresize/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_rescale_layer_call_and_return_conditional_losses_42945r
IdentityIdentity rescale/PartitionedCall:output:0*
T0*1
_output_shapes
:         ЦЦ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЦЦ:_ [
1
_output_shapes
:         ЦЦ
&
_user_specified_nameresize_input
л

√
@__inference_conv5_layer_call_and_return_conditional_losses_45266

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Й	
╬
3__inference_batch_normalization_layer_call_fn_44888

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_43049Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
├
Ц
&__inference_dense1_layer_call_fn_45368

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall╫
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_43568p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╜
C
'__inference_re_lu_4_layer_call_fn_45333

inputs
identity╢
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_43546i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
д

∙
@__inference_conv4_layer_call_and_return_conditional_losses_45165

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╦
Ы
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_45209

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_43373

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ю═
М 
G__inference_sequential_1_layer_call_and_return_conditional_losses_44813

inputs>
$conv1_conv2d_readvariableop_resource: 3
%conv1_biasadd_readvariableop_resource: 9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: >
$conv2_conv2d_readvariableop_resource:  3
%conv2_biasadd_readvariableop_resource: ;
-batch_normalization_1_readvariableop_resource: =
/batch_normalization_1_readvariableop_1_resource: L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: >
$conv3_conv2d_readvariableop_resource: @3
%conv3_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@>
$conv4_conv2d_readvariableop_resource:@@3
%conv4_biasadd_readvariableop_resource:@;
-batch_normalization_3_readvariableop_resource:@=
/batch_normalization_3_readvariableop_1_resource:@L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:@?
$conv5_conv2d_readvariableop_resource:@А4
%conv5_biasadd_readvariableop_resource:	А<
-batch_normalization_4_readvariableop_resource:	А>
/batch_normalization_4_readvariableop_1_resource:	АM
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	А9
%dense1_matmul_readvariableop_resource:
АА5
&dense1_biasadd_readvariableop_resource:	А8
$dense_matmul_readvariableop_resource:
А╞
4
%dense_biasadd_readvariableop_resource:	╞

identityИв"batch_normalization/AssignNewValueв$batch_normalization/AssignNewValue_1в3batch_normalization/FusedBatchNormV3/ReadVariableOpв5batch_normalization/FusedBatchNormV3/ReadVariableOp_1в"batch_normalization/ReadVariableOpв$batch_normalization/ReadVariableOp_1в$batch_normalization_1/AssignNewValueв&batch_normalization_1/AssignNewValue_1в5batch_normalization_1/FusedBatchNormV3/ReadVariableOpв7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_1/ReadVariableOpв&batch_normalization_1/ReadVariableOp_1в$batch_normalization_2/AssignNewValueв&batch_normalization_2/AssignNewValue_1в5batch_normalization_2/FusedBatchNormV3/ReadVariableOpв7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_2/ReadVariableOpв&batch_normalization_2/ReadVariableOp_1в$batch_normalization_3/AssignNewValueв&batch_normalization_3/AssignNewValue_1в5batch_normalization_3/FusedBatchNormV3/ReadVariableOpв7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_3/ReadVariableOpв&batch_normalization_3/ReadVariableOp_1в$batch_normalization_4/AssignNewValueв&batch_normalization_4/AssignNewValue_1в5batch_normalization_4/FusedBatchNormV3/ReadVariableOpв7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_4/ReadVariableOpв&batch_normalization_4/ReadVariableOp_1вconv1/BiasAdd/ReadVariableOpвconv1/Conv2D/ReadVariableOpвconv2/BiasAdd/ReadVariableOpвconv2/Conv2D/ReadVariableOpвconv3/BiasAdd/ReadVariableOpвconv3/Conv2D/ReadVariableOpвconv4/BiasAdd/ReadVariableOpвconv4/Conv2D/ReadVariableOpвconv5/BiasAdd/ReadVariableOpвconv5/Conv2D/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense1/BiasAdd/ReadVariableOpвdense1/MatMul/ReadVariableOpn
sequential/resize/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Ц   Ц   ┐
'sequential/resize/resize/ResizeBilinearResizeBilinearinputs&sequential/resize/resize/size:output:0*
T0*1
_output_shapes
:         ЦЦ*
half_pixel_centers(^
sequential/rescale/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;`
sequential/rescale/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ╖
sequential/rescale/mulMul8sequential/resize/resize/ResizeBilinear:resized_images:0"sequential/rescale/Cast/x:output:0*
T0*1
_output_shapes
:         ЦЦЭ
sequential/rescale/addAddV2sequential/rescale/mul:z:0$sequential/rescale/Cast_1/x:output:0*
T0*1
_output_shapes
:         ЦЦИ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╝
conv1/Conv2DConv2Dsequential/rescale/add:z:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ФФ *
paddingVALID*
strides
~
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0С
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ФФ К
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0О
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0м
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0░
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0║
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv1/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ФФ : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ц
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(x

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         ФФ з
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*/
_output_shapes
:         JJ *
ksize
*
paddingVALID*
strides
И
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0╛
conv2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         HH *
paddingVALID*
strides
~
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0П
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         HH О
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype0Т
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0┬
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         HH : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         HH л
max_pooling2d_1/MaxPoolMaxPoolre_lu_1/Relu:activations:0*/
_output_shapes
:         $$ *
ksize
*
paddingVALID*
strides
И
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0└
conv3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ""@*
paddingVALID*
strides
~
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ""@О
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0Т
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0┬
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv3/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         ""@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         ""@л
max_pooling2d_2/MaxPoolMaxPoolre_lu_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
И
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0└
conv4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
~
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @О
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0Т
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0┬
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv4/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
re_lu_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @л
max_pooling2d_3/MaxPoolMaxPoolre_lu_3/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
Й
conv5/Conv2D/ReadVariableOpReadVariableOp$conv5_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0┴
conv5/Conv2DConv2D max_pooling2d_3/MaxPool:output:0#conv5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides

conv5/BiasAdd/ReadVariableOpReadVariableOp%conv5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Р
conv5/BiasAddBiasAddconv5/Conv2D:output:0$conv5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АП
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:А*
dtype0У
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:А*
dtype0▒
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╡
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╟
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv5/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape({
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         Ам
max_pooling2d_4/MaxPoolMaxPoolre_lu_4/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       З
flatten/ReshapeReshape max_pooling2d_4/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         АД
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0К
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АБ
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0М
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А_
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:         АZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?И
dropout/dropout/MulMuldense1/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:         А^
dropout/dropout/ShapeShapedense1/Relu:activations:0*
T0*
_output_shapes
:Э
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>┐
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ┤
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*(
_output_shapes
:         АВ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
А╞
*
dtype0С
dense/MatMulMatMul!dropout/dropout/SelectV2:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╞

dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:╞
*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╞
c
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*(
_output_shapes
:         ╞
g
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:         ╞
┴
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp^conv5/BiasAdd/ReadVariableOp^conv5/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2<
conv5/BiasAdd/ReadVariableOpconv5/BiasAdd/ReadVariableOp2:
conv5/Conv2D/ReadVariableOpconv5/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
Ч	
╘
5__inference_batch_normalization_4_layer_call_fn_45279

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43322К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
ц
^
B__inference_re_lu_1_layer_call_and_return_conditional_losses_43447

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         HH b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         HH "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         HH :W S
/
_output_shapes
:         HH 
 
_user_specified_nameinputs
▒б
■&
__inference__traced_save_45737
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop+
'savev2_conv3_kernel_read_readvariableop)
%savev2_conv3_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop+
'savev2_conv4_kernel_read_readvariableop)
%savev2_conv4_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop+
'savev2_conv5_kernel_read_readvariableop)
%savev2_conv5_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop2
.savev2_adam_m_conv1_kernel_read_readvariableop2
.savev2_adam_v_conv1_kernel_read_readvariableop0
,savev2_adam_m_conv1_bias_read_readvariableop0
,savev2_adam_v_conv1_bias_read_readvariableop?
;savev2_adam_m_batch_normalization_gamma_read_readvariableop?
;savev2_adam_v_batch_normalization_gamma_read_readvariableop>
:savev2_adam_m_batch_normalization_beta_read_readvariableop>
:savev2_adam_v_batch_normalization_beta_read_readvariableop2
.savev2_adam_m_conv2_kernel_read_readvariableop2
.savev2_adam_v_conv2_kernel_read_readvariableop0
,savev2_adam_m_conv2_bias_read_readvariableop0
,savev2_adam_v_conv2_bias_read_readvariableopA
=savev2_adam_m_batch_normalization_1_gamma_read_readvariableopA
=savev2_adam_v_batch_normalization_1_gamma_read_readvariableop@
<savev2_adam_m_batch_normalization_1_beta_read_readvariableop@
<savev2_adam_v_batch_normalization_1_beta_read_readvariableop2
.savev2_adam_m_conv3_kernel_read_readvariableop2
.savev2_adam_v_conv3_kernel_read_readvariableop0
,savev2_adam_m_conv3_bias_read_readvariableop0
,savev2_adam_v_conv3_bias_read_readvariableopA
=savev2_adam_m_batch_normalization_2_gamma_read_readvariableopA
=savev2_adam_v_batch_normalization_2_gamma_read_readvariableop@
<savev2_adam_m_batch_normalization_2_beta_read_readvariableop@
<savev2_adam_v_batch_normalization_2_beta_read_readvariableop2
.savev2_adam_m_conv4_kernel_read_readvariableop2
.savev2_adam_v_conv4_kernel_read_readvariableop0
,savev2_adam_m_conv4_bias_read_readvariableop0
,savev2_adam_v_conv4_bias_read_readvariableopA
=savev2_adam_m_batch_normalization_3_gamma_read_readvariableopA
=savev2_adam_v_batch_normalization_3_gamma_read_readvariableop@
<savev2_adam_m_batch_normalization_3_beta_read_readvariableop@
<savev2_adam_v_batch_normalization_3_beta_read_readvariableop2
.savev2_adam_m_conv5_kernel_read_readvariableop2
.savev2_adam_v_conv5_kernel_read_readvariableop0
,savev2_adam_m_conv5_bias_read_readvariableop0
,savev2_adam_v_conv5_bias_read_readvariableopA
=savev2_adam_m_batch_normalization_4_gamma_read_readvariableopA
=savev2_adam_v_batch_normalization_4_gamma_read_readvariableop@
<savev2_adam_m_batch_normalization_4_beta_read_readvariableop@
<savev2_adam_v_batch_normalization_4_beta_read_readvariableop3
/savev2_adam_m_dense1_kernel_read_readvariableop3
/savev2_adam_v_dense1_kernel_read_readvariableop1
-savev2_adam_m_dense1_bias_read_readvariableop1
-savev2_adam_v_dense1_bias_read_readvariableop2
.savev2_adam_m_dense_kernel_read_readvariableop2
.savev2_adam_v_dense_kernel_read_readvariableop0
,savev2_adam_m_dense_bias_read_readvariableop0
,savev2_adam_v_dense_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Э&
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*╞%
value╝%B╣%YB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHв
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*╟
value╜B║YB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B у%
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop'savev2_conv3_kernel_read_readvariableop%savev2_conv3_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop'savev2_conv4_kernel_read_readvariableop%savev2_conv4_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop'savev2_conv5_kernel_read_readvariableop%savev2_conv5_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop.savev2_adam_m_conv1_kernel_read_readvariableop.savev2_adam_v_conv1_kernel_read_readvariableop,savev2_adam_m_conv1_bias_read_readvariableop,savev2_adam_v_conv1_bias_read_readvariableop;savev2_adam_m_batch_normalization_gamma_read_readvariableop;savev2_adam_v_batch_normalization_gamma_read_readvariableop:savev2_adam_m_batch_normalization_beta_read_readvariableop:savev2_adam_v_batch_normalization_beta_read_readvariableop.savev2_adam_m_conv2_kernel_read_readvariableop.savev2_adam_v_conv2_kernel_read_readvariableop,savev2_adam_m_conv2_bias_read_readvariableop,savev2_adam_v_conv2_bias_read_readvariableop=savev2_adam_m_batch_normalization_1_gamma_read_readvariableop=savev2_adam_v_batch_normalization_1_gamma_read_readvariableop<savev2_adam_m_batch_normalization_1_beta_read_readvariableop<savev2_adam_v_batch_normalization_1_beta_read_readvariableop.savev2_adam_m_conv3_kernel_read_readvariableop.savev2_adam_v_conv3_kernel_read_readvariableop,savev2_adam_m_conv3_bias_read_readvariableop,savev2_adam_v_conv3_bias_read_readvariableop=savev2_adam_m_batch_normalization_2_gamma_read_readvariableop=savev2_adam_v_batch_normalization_2_gamma_read_readvariableop<savev2_adam_m_batch_normalization_2_beta_read_readvariableop<savev2_adam_v_batch_normalization_2_beta_read_readvariableop.savev2_adam_m_conv4_kernel_read_readvariableop.savev2_adam_v_conv4_kernel_read_readvariableop,savev2_adam_m_conv4_bias_read_readvariableop,savev2_adam_v_conv4_bias_read_readvariableop=savev2_adam_m_batch_normalization_3_gamma_read_readvariableop=savev2_adam_v_batch_normalization_3_gamma_read_readvariableop<savev2_adam_m_batch_normalization_3_beta_read_readvariableop<savev2_adam_v_batch_normalization_3_beta_read_readvariableop.savev2_adam_m_conv5_kernel_read_readvariableop.savev2_adam_v_conv5_kernel_read_readvariableop,savev2_adam_m_conv5_bias_read_readvariableop,savev2_adam_v_conv5_bias_read_readvariableop=savev2_adam_m_batch_normalization_4_gamma_read_readvariableop=savev2_adam_v_batch_normalization_4_gamma_read_readvariableop<savev2_adam_m_batch_normalization_4_beta_read_readvariableop<savev2_adam_v_batch_normalization_4_beta_read_readvariableop/savev2_adam_m_dense1_kernel_read_readvariableop/savev2_adam_v_dense1_kernel_read_readvariableop-savev2_adam_m_dense1_bias_read_readvariableop-savev2_adam_v_dense1_bias_read_readvariableop.savev2_adam_m_dense_kernel_read_readvariableop.savev2_adam_v_dense_kernel_read_readvariableop,savev2_adam_m_dense_bias_read_readvariableop,savev2_adam_v_dense_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *g
dtypes]
[2Y	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*¤
_input_shapesы
ш: : : : : : : :  : : : : : : @:@:@:@:@:@:@@:@:@:@:@:@:@А:А:А:А:А:А:
АА:А:
А╞
:╞
: : : : : : : : : : :  :  : : : : : : : @: @:@:@:@:@:@:@:@@:@@:@:@:@:@:@:@:@А:@А:А:А:А:А:А:А:
АА:
АА:А:А:
А╞
:
А╞
:╞
:╞
: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:! 

_output_shapes	
:А:&!"
 
_output_shapes
:
А╞
:!"

_output_shapes	
:╞
:#

_output_shapes
: :$

_output_shapes
: :,%(
&
_output_shapes
: :,&(
&
_output_shapes
: : '

_output_shapes
: : (

_output_shapes
: : )

_output_shapes
: : *

_output_shapes
: : +

_output_shapes
: : ,

_output_shapes
: :,-(
&
_output_shapes
:  :,.(
&
_output_shapes
:  : /

_output_shapes
: : 0

_output_shapes
: : 1

_output_shapes
: : 2

_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: :,5(
&
_output_shapes
: @:,6(
&
_output_shapes
: @: 7

_output_shapes
:@: 8

_output_shapes
:@: 9

_output_shapes
:@: :

_output_shapes
:@: ;

_output_shapes
:@: <

_output_shapes
:@:,=(
&
_output_shapes
:@@:,>(
&
_output_shapes
:@@: ?

_output_shapes
:@: @

_output_shapes
:@: A

_output_shapes
:@: B

_output_shapes
:@: C

_output_shapes
:@: D

_output_shapes
:@:-E)
'
_output_shapes
:@А:-F)
'
_output_shapes
:@А:!G

_output_shapes	
:А:!H

_output_shapes	
:А:!I

_output_shapes	
:А:!J

_output_shapes	
:А:!K

_output_shapes	
:А:!L

_output_shapes	
:А:&M"
 
_output_shapes
:
АА:&N"
 
_output_shapes
:
АА:!O

_output_shapes	
:А:!P

_output_shapes	
:А:&Q"
 
_output_shapes
:
А╞
:&R"
 
_output_shapes
:
А╞
:!S

_output_shapes	
:╞
:!T

_output_shapes	
:╞
:U

_output_shapes
: :V

_output_shapes
: :W

_output_shapes
: :X

_output_shapes
: :Y

_output_shapes
: 
╢
K
/__inference_max_pooling2d_2_layer_call_fn_45141

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_43221Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_43297

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
т
Ъ
%__inference_conv2_layer_call_fn_44953

inputs!
unknown:  
	unknown_0: 
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         HH *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_43427w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         HH `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         JJ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         JJ 
 
_user_specified_nameinputs
╣
C
'__inference_re_lu_2_layer_call_fn_45131

inputs
identity╡
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ""@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_43480h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         ""@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         ""@:W S
/
_output_shapes
:         ""@
 
_user_specified_nameinputs
╢
K
/__inference_max_pooling2d_3_layer_call_fn_45242

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_43297Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Е
┐
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_45025

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
и

Ї
@__inference_dense_layer_call_and_return_conditional_losses_43592

inputs2
matmul_readvariableop_resource:
А╞
.
biasadd_readvariableop_resource:	╞

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А╞
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╞
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╞
*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╞
W
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:         ╞
a
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:         ╞
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ъ
Ъ
%__inference_conv1_layer_call_fn_44852

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ФФ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_43394y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ФФ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЦЦ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
Х	
╘
5__inference_batch_normalization_4_layer_call_fn_45292

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43353К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ё
a
E__inference_sequential_layer_call_and_return_conditional_losses_44833

inputs
identityc
resize/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Ц   Ц   й
resize/resize/ResizeBilinearResizeBilinearinputsresize/resize/size:output:0*
T0*1
_output_shapes
:         ЦЦ*
half_pixel_centers(S
rescale/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;U
rescale/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
rescale/mulMul-resize/resize/ResizeBilinear:resized_images:0rescale/Cast/x:output:0*
T0*1
_output_shapes
:         ЦЦ|
rescale/addAddV2rescale/mul:z:0rescale/Cast_1/x:output:0*
T0*1
_output_shapes
:         ЦЦa
IdentityIdentityrescale/add:z:0*
T0*1
_output_shapes
:         ЦЦ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЦЦ:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
╞
Э
#__inference_signature_wrapper_44386
sequential_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@%

unknown_23:@А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:
АА

unknown_30:	А

unknown_31:
А╞


unknown_32:	╞

identityИвStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╞
*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_42922p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╞
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:         ЦЦ
*
_user_specified_namesequential_input
д

∙
@__inference_conv2_layer_call_and_return_conditional_losses_43427

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         HH *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         HH g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         HH w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         JJ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         JJ 
 
_user_specified_nameinputs
┘
L
*__inference_sequential_layer_call_fn_42951
resize_input
identity└
PartitionedCallPartitionedCallresize_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_42948j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ЦЦ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЦЦ:_ [
1
_output_shapes
:         ЦЦ
&
_user_specified_nameresize_input
┌
g
E__inference_sequential_layer_call_and_return_conditional_losses_42990
resize_input
identity├
resize/PartitionedCallPartitionedCallresize_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_resize_layer_call_and_return_conditional_losses_42935╪
rescale/PartitionedCallPartitionedCallresize/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_rescale_layer_call_and_return_conditional_losses_42945r
IdentityIdentity rescale/PartitionedCall:output:0*
T0*1
_output_shapes
:         ЦЦ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЦЦ:_ [
1
_output_shapes
:         ЦЦ
&
_user_specified_nameresize_input
Е
┐
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_45227

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╞
^
B__inference_flatten_layer_call_and_return_conditional_losses_43555

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
█
Я
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_45310

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
л
]
A__inference_resize_layer_call_and_return_conditional_losses_45437

inputs
identity\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Ц   Ц   Ы
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*1
_output_shapes
:         ЦЦ*
half_pixel_centers(x
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:         ЦЦ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЦЦ:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
Ё
a
E__inference_sequential_layer_call_and_return_conditional_losses_44843

inputs
identityc
resize/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Ц   Ц   й
resize/resize/ResizeBilinearResizeBilinearinputsresize/resize/size:output:0*
T0*1
_output_shapes
:         ЦЦ*
half_pixel_centers(S
rescale/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;U
rescale/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
rescale/mulMul-resize/resize/ResizeBilinear:resized_images:0rescale/Cast/x:output:0*
T0*1
_output_shapes
:         ЦЦ|
rescale/addAddV2rescale/mul:z:0rescale/Cast_1/x:output:0*
T0*1
_output_shapes
:         ЦЦa
IdentityIdentityrescale/add:z:0*
T0*1
_output_shapes
:         ЦЦ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЦЦ:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
д

∙
@__inference_conv4_layer_call_and_return_conditional_losses_43493

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Е
┐
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43201

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
┘
L
*__inference_sequential_layer_call_fn_42984
resize_input
identity└
PartitionedCallPartitionedCallresize_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_42976j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ЦЦ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЦЦ:_ [
1
_output_shapes
:         ЦЦ
&
_user_specified_nameresize_input
н
C
'__inference_flatten_layer_call_fn_45353

inputs
identityо
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_43555a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
д

ї
A__inference_dense1_layer_call_and_return_conditional_losses_43568

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
П	
╨
5__inference_batch_normalization_3_layer_call_fn_45178

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43246Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╞
^
B__inference_flatten_layer_call_and_return_conditional_losses_45359

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Х
├
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43353

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
█
Я
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43322

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╚
a
E__inference_sequential_layer_call_and_return_conditional_losses_42976

inputs
identity╜
resize/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_resize_layer_call_and_return_conditional_losses_42935╪
rescale/PartitionedCallPartitionedCallresize/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_rescale_layer_call_and_return_conditional_losses_42945r
IdentityIdentity rescale/PartitionedCall:output:0*
T0*1
_output_shapes
:         ЦЦ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЦЦ:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs"Ж
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┼
serving_default▒
W
sequential_inputC
"serving_default_sequential_input:0         ЦЦ:
dense1
StatefulPartitionedCall:0         ╞
tensorflow/serving/predict:╪Р
В
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
layer-16
layer-17
layer_with_weights-8
layer-18
layer_with_weights-9
layer-19
layer-20
layer-21
layer-22
layer_with_weights-10
layer-23
layer-24
layer_with_weights-11
layer-25
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!_default_save_signature
"	optimizer
#
signatures"
_tf_keras_sequential
─
$layer-0
%layer-1
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_sequential
"
_tf_keras_input_layer
▌
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op"
_tf_keras_layer
ъ
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;axis
	<gamma
=beta
>moving_mean
?moving_variance"
_tf_keras_layer
е
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
е
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses

Rkernel
Sbias
 T_jit_compiled_convolution_op"
_tf_keras_layer
ъ
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[axis
	\gamma
]beta
^moving_mean
_moving_variance"
_tf_keras_layer
е
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_layer
е
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

rkernel
sbias
 t_jit_compiled_convolution_op"
_tf_keras_layer
ъ
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
{axis
	|gamma
}beta
~moving_mean
moving_variance"
_tf_keras_layer
л
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
К__call__
+Л&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses
Тkernel
	Уbias
!Ф_jit_compiled_convolution_op"
_tf_keras_layer
ї
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses
	Ыaxis

Ьgamma
	Эbeta
Юmoving_mean
Яmoving_variance"
_tf_keras_layer
л
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses"
_tf_keras_layer
л
ж	variables
зtrainable_variables
иregularization_losses
й	keras_api
к__call__
+л&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
м	variables
нtrainable_variables
оregularization_losses
п	keras_api
░__call__
+▒&call_and_return_all_conditional_losses
▓kernel
	│bias
!┤_jit_compiled_convolution_op"
_tf_keras_layer
ї
╡	variables
╢trainable_variables
╖regularization_losses
╕	keras_api
╣__call__
+║&call_and_return_all_conditional_losses
	╗axis

╝gamma
	╜beta
╛moving_mean
┐moving_variance"
_tf_keras_layer
л
└	variables
┴trainable_variables
┬regularization_losses
├	keras_api
─__call__
+┼&call_and_return_all_conditional_losses"
_tf_keras_layer
л
╞	variables
╟trainable_variables
╚regularization_losses
╔	keras_api
╩__call__
+╦&call_and_return_all_conditional_losses"
_tf_keras_layer
л
╠	variables
═trainable_variables
╬regularization_losses
╧	keras_api
╨__call__
+╤&call_and_return_all_conditional_losses"
_tf_keras_layer
├
╥	variables
╙trainable_variables
╘regularization_losses
╒	keras_api
╓__call__
+╫&call_and_return_all_conditional_losses
╪kernel
	┘bias"
_tf_keras_layer
├
┌	variables
█trainable_variables
▄regularization_losses
▌	keras_api
▐__call__
+▀&call_and_return_all_conditional_losses
р_random_generator"
_tf_keras_layer
├
с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
х__call__
+ц&call_and_return_all_conditional_losses
чkernel
	шbias"
_tf_keras_layer
╢
20
31
<2
=3
>4
?5
R6
S7
\8
]9
^10
_11
r12
s13
|14
}15
~16
17
Т18
У19
Ь20
Э21
Ю22
Я23
▓24
│25
╝26
╜27
╛28
┐29
╪30
┘31
ч32
ш33"
trackable_list_wrapper
т
20
31
<2
=3
R4
S5
\6
]7
r8
s9
|10
}11
Т12
У13
Ь14
Э15
▓16
│17
╝18
╜19
╪20
┘21
ч22
ш23"
trackable_list_wrapper
 "
trackable_list_wrapper
╧
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
!_default_save_signature
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
э
юtrace_0
яtrace_1
Ёtrace_2
ёtrace_32·
,__inference_sequential_1_layer_call_fn_43670
,__inference_sequential_1_layer_call_fn_44459
,__inference_sequential_1_layer_call_fn_44532
,__inference_sequential_1_layer_call_fn_44115┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zюtrace_0zяtrace_1zЁtrace_2zёtrace_3
┘
Єtrace_0
єtrace_1
Їtrace_2
їtrace_32ц
G__inference_sequential_1_layer_call_and_return_conditional_losses_44669
G__inference_sequential_1_layer_call_and_return_conditional_losses_44813
G__inference_sequential_1_layer_call_and_return_conditional_losses_44212
G__inference_sequential_1_layer_call_and_return_conditional_losses_44309┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЄtrace_0zєtrace_1zЇtrace_2zїtrace_3
╘B╤
 __inference__wrapped_model_42922sequential_input"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
г
Ў
_variables
ў_iterations
°_learning_rate
∙_index_dict
·
_momentums
√_velocities
№_update_step_xla"
experimentalOptimizer
-
¤serving_default"
signature_map
л
■	variables
 trainable_variables
Аregularization_losses
Б	keras_api
В__call__
+Г&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
И__call__
+Й&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
х
Пtrace_0
Рtrace_1
Сtrace_2
Тtrace_32Є
*__inference_sequential_layer_call_fn_42951
*__inference_sequential_layer_call_fn_44818
*__inference_sequential_layer_call_fn_44823
*__inference_sequential_layer_call_fn_42984┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zПtrace_0zРtrace_1zСtrace_2zТtrace_3
╤
Уtrace_0
Фtrace_1
Хtrace_2
Цtrace_32▐
E__inference_sequential_layer_call_and_return_conditional_losses_44833
E__inference_sequential_layer_call_and_return_conditional_losses_44843
E__inference_sequential_layer_call_and_return_conditional_losses_42990
E__inference_sequential_layer_call_and_return_conditional_losses_42996┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zУtrace_0zФtrace_1zХtrace_2zЦtrace_3
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
ы
Ьtrace_02╠
%__inference_conv1_layer_call_fn_44852в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЬtrace_0
Ж
Эtrace_02ч
@__inference_conv1_layer_call_and_return_conditional_losses_44862в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЭtrace_0
&:$ 2conv1/kernel
: 2
conv1/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
<
<0
=1
>2
?3"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
█
гtrace_0
дtrace_12а
3__inference_batch_normalization_layer_call_fn_44875
3__inference_batch_normalization_layer_call_fn_44888│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zгtrace_0zдtrace_1
С
еtrace_0
жtrace_12╓
N__inference_batch_normalization_layer_call_and_return_conditional_losses_44906
N__inference_batch_normalization_layer_call_and_return_conditional_losses_44924│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zеtrace_0zжtrace_1
 "
trackable_list_wrapper
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
зnon_trainable_variables
иlayers
йmetrics
 кlayer_regularization_losses
лlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
ы
мtrace_02╠
%__inference_re_lu_layer_call_fn_44929в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zмtrace_0
Ж
нtrace_02ч
@__inference_re_lu_layer_call_and_return_conditional_losses_44934в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zнtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
оnon_trainable_variables
пlayers
░metrics
 ▒layer_regularization_losses
▓layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
є
│trace_02╘
-__inference_max_pooling2d_layer_call_fn_44939в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z│trace_0
О
┤trace_02я
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_44944в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┤trace_0
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╡non_trainable_variables
╢layers
╖metrics
 ╕layer_regularization_losses
╣layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
ы
║trace_02╠
%__inference_conv2_layer_call_fn_44953в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z║trace_0
Ж
╗trace_02ч
@__inference_conv2_layer_call_and_return_conditional_losses_44963в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╗trace_0
&:$  2conv2/kernel
: 2
conv2/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
<
\0
]1
^2
_3"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╝non_trainable_variables
╜layers
╛metrics
 ┐layer_regularization_losses
└layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
▀
┴trace_0
┬trace_12д
5__inference_batch_normalization_1_layer_call_fn_44976
5__inference_batch_normalization_1_layer_call_fn_44989│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┴trace_0z┬trace_1
Х
├trace_0
─trace_12┌
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_45007
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_45025│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z├trace_0z─trace_1
 "
trackable_list_wrapper
):' 2batch_normalization_1/gamma
(:& 2batch_normalization_1/beta
1:/  (2!batch_normalization_1/moving_mean
5:3  (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┼non_trainable_variables
╞layers
╟metrics
 ╚layer_regularization_losses
╔layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
э
╩trace_02╬
'__inference_re_lu_1_layer_call_fn_45030в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╩trace_0
И
╦trace_02щ
B__inference_re_lu_1_layer_call_and_return_conditional_losses_45035в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╦trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
ї
╤trace_02╓
/__inference_max_pooling2d_1_layer_call_fn_45040в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╤trace_0
Р
╥trace_02ё
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_45045в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╥trace_0
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
ы
╪trace_02╠
%__inference_conv3_layer_call_fn_45054в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╪trace_0
Ж
┘trace_02ч
@__inference_conv3_layer_call_and_return_conditional_losses_45064в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┘trace_0
&:$ @2conv3/kernel
:@2
conv3/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
<
|0
}1
~2
3"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┌non_trainable_variables
█layers
▄metrics
 ▌layer_regularization_losses
▐layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
▀
▀trace_0
рtrace_12д
5__inference_batch_normalization_2_layer_call_fn_45077
5__inference_batch_normalization_2_layer_call_fn_45090│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▀trace_0zрtrace_1
Х
сtrace_0
тtrace_12┌
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_45108
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_45126│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zсtrace_0zтtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
А	variables
Бtrainable_variables
Вregularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
э
шtrace_02╬
'__inference_re_lu_2_layer_call_fn_45131в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zшtrace_0
И
щtrace_02щ
B__inference_re_lu_2_layer_call_and_return_conditional_losses_45136в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zщtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
Ж	variables
Зtrainable_variables
Иregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
ї
яtrace_02╓
/__inference_max_pooling2d_2_layer_call_fn_45141в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zяtrace_0
Р
Ёtrace_02ё
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_45146в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЁtrace_0
0
Т0
У1"
trackable_list_wrapper
0
Т0
У1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
ёnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
ы
Ўtrace_02╠
%__inference_conv4_layer_call_fn_45155в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЎtrace_0
Ж
ўtrace_02ч
@__inference_conv4_layer_call_and_return_conditional_losses_45165в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zўtrace_0
&:$@@2conv4/kernel
:@2
conv4/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
@
Ь0
Э1
Ю2
Я3"
trackable_list_wrapper
0
Ь0
Э1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
°non_trainable_variables
∙layers
·metrics
 √layer_regularization_losses
№layer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
▀
¤trace_0
■trace_12д
5__inference_batch_normalization_3_layer_call_fn_45178
5__inference_batch_normalization_3_layer_call_fn_45191│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z¤trace_0z■trace_1
Х
 trace_0
Аtrace_12┌
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_45209
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_45227│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z trace_0zАtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_3/gamma
(:&@2batch_normalization_3/beta
1:/@ (2!batch_normalization_3/moving_mean
5:3@ (2%batch_normalization_3/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
э
Жtrace_02╬
'__inference_re_lu_3_layer_call_fn_45232в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЖtrace_0
И
Зtrace_02щ
B__inference_re_lu_3_layer_call_and_return_conditional_losses_45237в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЗtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
ж	variables
зtrainable_variables
иregularization_losses
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
ї
Нtrace_02╓
/__inference_max_pooling2d_3_layer_call_fn_45242в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zНtrace_0
Р
Оtrace_02ё
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_45247в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zОtrace_0
0
▓0
│1"
trackable_list_wrapper
0
▓0
│1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
м	variables
нtrainable_variables
оregularization_losses
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
ы
Фtrace_02╠
%__inference_conv5_layer_call_fn_45256в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zФtrace_0
Ж
Хtrace_02ч
@__inference_conv5_layer_call_and_return_conditional_losses_45266в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zХtrace_0
':%@А2conv5/kernel
:А2
conv5/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
@
╝0
╜1
╛2
┐3"
trackable_list_wrapper
0
╝0
╜1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
╡	variables
╢trainable_variables
╖regularization_losses
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
▀
Ыtrace_0
Ьtrace_12д
5__inference_batch_normalization_4_layer_call_fn_45279
5__inference_batch_normalization_4_layer_call_fn_45292│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЫtrace_0zЬtrace_1
Х
Эtrace_0
Юtrace_12┌
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_45310
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_45328│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЭtrace_0zЮtrace_1
 "
trackable_list_wrapper
*:(А2batch_normalization_4/gamma
):'А2batch_normalization_4/beta
2:0А (2!batch_normalization_4/moving_mean
6:4А (2%batch_normalization_4/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
└	variables
┴trainable_variables
┬regularization_losses
─__call__
+┼&call_and_return_all_conditional_losses
'┼"call_and_return_conditional_losses"
_generic_user_object
э
дtrace_02╬
'__inference_re_lu_4_layer_call_fn_45333в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zдtrace_0
И
еtrace_02щ
B__inference_re_lu_4_layer_call_and_return_conditional_losses_45338в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zеtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
╞	variables
╟trainable_variables
╚regularization_losses
╩__call__
+╦&call_and_return_all_conditional_losses
'╦"call_and_return_conditional_losses"
_generic_user_object
ї
лtrace_02╓
/__inference_max_pooling2d_4_layer_call_fn_45343в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zлtrace_0
Р
мtrace_02ё
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_45348в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zмtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
нnon_trainable_variables
оlayers
пmetrics
 ░layer_regularization_losses
▒layer_metrics
╠	variables
═trainable_variables
╬regularization_losses
╨__call__
+╤&call_and_return_all_conditional_losses
'╤"call_and_return_conditional_losses"
_generic_user_object
э
▓trace_02╬
'__inference_flatten_layer_call_fn_45353в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▓trace_0
И
│trace_02щ
B__inference_flatten_layer_call_and_return_conditional_losses_45359в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z│trace_0
0
╪0
┘1"
trackable_list_wrapper
0
╪0
┘1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┤non_trainable_variables
╡layers
╢metrics
 ╖layer_regularization_losses
╕layer_metrics
╥	variables
╙trainable_variables
╘regularization_losses
╓__call__
+╫&call_and_return_all_conditional_losses
'╫"call_and_return_conditional_losses"
_generic_user_object
ь
╣trace_02═
&__inference_dense1_layer_call_fn_45368в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╣trace_0
З
║trace_02ш
A__inference_dense1_layer_call_and_return_conditional_losses_45379в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z║trace_0
!:
АА2dense1/kernel
:А2dense1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╗non_trainable_variables
╝layers
╜metrics
 ╛layer_regularization_losses
┐layer_metrics
┌	variables
█trainable_variables
▄regularization_losses
▐__call__
+▀&call_and_return_all_conditional_losses
'▀"call_and_return_conditional_losses"
_generic_user_object
├
└trace_0
┴trace_12И
'__inference_dropout_layer_call_fn_45384
'__inference_dropout_layer_call_fn_45389│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z└trace_0z┴trace_1
∙
┬trace_0
├trace_12╛
B__inference_dropout_layer_call_and_return_conditional_losses_45394
B__inference_dropout_layer_call_and_return_conditional_losses_45406│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┬trace_0z├trace_1
"
_generic_user_object
0
ч0
ш1"
trackable_list_wrapper
0
ч0
ш1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
─non_trainable_variables
┼layers
╞metrics
 ╟layer_regularization_losses
╚layer_metrics
с	variables
тtrainable_variables
уregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
ы
╔trace_02╠
%__inference_dense_layer_call_fn_45415в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╔trace_0
Ж
╩trace_02ч
@__inference_dense_layer_call_and_return_conditional_losses_45426в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╩trace_0
 :
А╞
2dense/kernel
:╞
2
dense/bias
j
>0
?1
^2
_3
~4
5
Ю6
Я7
╛8
┐9"
trackable_list_wrapper
ц
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25"
trackable_list_wrapper
0
╦0
╠1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЗBД
,__inference_sequential_1_layer_call_fn_43670sequential_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
,__inference_sequential_1_layer_call_fn_44459inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
,__inference_sequential_1_layer_call_fn_44532inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЗBД
,__inference_sequential_1_layer_call_fn_44115sequential_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
G__inference_sequential_1_layer_call_and_return_conditional_losses_44669inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
G__inference_sequential_1_layer_call_and_return_conditional_losses_44813inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
вBЯ
G__inference_sequential_1_layer_call_and_return_conditional_losses_44212sequential_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
вBЯ
G__inference_sequential_1_layer_call_and_return_conditional_losses_44309sequential_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╧
ў0
═1
╬2
╧3
╨4
╤5
╥6
╙7
╘8
╒9
╓10
╫11
╪12
┘13
┌14
█15
▄16
▌17
▐18
▀19
р20
с21
т22
у23
ф24
х25
ц26
ч27
ш28
щ29
ъ30
ы31
ь32
э33
ю34
я35
Ё36
ё37
Є38
є39
Ї40
ї41
Ў42
ў43
°44
∙45
·46
√47
№48"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
ю
═0
╧1
╤2
╙3
╒4
╫5
┘6
█7
▌8
▀9
с10
у11
х12
ч13
щ14
ы15
э16
я17
ё18
є19
ї20
ў21
∙22
√23"
trackable_list_wrapper
ю
╬0
╨1
╥2
╘3
╓4
╪5
┌6
▄7
▐8
р9
т10
ф11
ц12
ш13
ъ14
ь15
ю16
Ё17
Є18
Ї19
Ў20
°21
·22
№23"
trackable_list_wrapper
┐2╝╣
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
╙B╨
#__inference_signature_wrapper_44386sequential_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
¤non_trainable_variables
■layers
 metrics
 Аlayer_regularization_losses
Бlayer_metrics
■	variables
 trainable_variables
Аregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
ь
Вtrace_02═
&__inference_resize_layer_call_fn_45431в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zВtrace_0
З
Гtrace_02ш
A__inference_resize_layer_call_and_return_conditional_losses_45437в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zГtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
э
Йtrace_02╬
'__inference_rescale_layer_call_fn_45442в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЙtrace_0
И
Кtrace_02щ
B__inference_rescale_layer_call_and_return_conditional_losses_45450в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zКtrace_0
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
БB■
*__inference_sequential_layer_call_fn_42951resize_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
*__inference_sequential_layer_call_fn_44818inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
*__inference_sequential_layer_call_fn_44823inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
БB■
*__inference_sequential_layer_call_fn_42984resize_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЦBУ
E__inference_sequential_layer_call_and_return_conditional_losses_44833inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЦBУ
E__inference_sequential_layer_call_and_return_conditional_losses_44843inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЬBЩ
E__inference_sequential_layer_call_and_return_conditional_losses_42990resize_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЬBЩ
E__inference_sequential_layer_call_and_return_conditional_losses_42996resize_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
%__inference_conv1_layer_call_fn_44852inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
@__inference_conv1_layer_call_and_return_conditional_losses_44862inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
°Bї
3__inference_batch_normalization_layer_call_fn_44875inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
3__inference_batch_normalization_layer_call_fn_44888inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
УBР
N__inference_batch_normalization_layer_call_and_return_conditional_losses_44906inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
УBР
N__inference_batch_normalization_layer_call_and_return_conditional_losses_44924inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
%__inference_re_lu_layer_call_fn_44929inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
@__inference_re_lu_layer_call_and_return_conditional_losses_44934inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сB▐
-__inference_max_pooling2d_layer_call_fn_44939inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_44944inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
%__inference_conv2_layer_call_fn_44953inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
@__inference_conv2_layer_call_and_return_conditional_losses_44963inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
·Bў
5__inference_batch_normalization_1_layer_call_fn_44976inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
5__inference_batch_normalization_1_layer_call_fn_44989inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_45007inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_45025inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
█B╪
'__inference_re_lu_1_layer_call_fn_45030inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_re_lu_1_layer_call_and_return_conditional_losses_45035inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBр
/__inference_max_pooling2d_1_layer_call_fn_45040inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_45045inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
%__inference_conv3_layer_call_fn_45054inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
@__inference_conv3_layer_call_and_return_conditional_losses_45064inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
·Bў
5__inference_batch_normalization_2_layer_call_fn_45077inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
5__inference_batch_normalization_2_layer_call_fn_45090inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_45108inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_45126inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
█B╪
'__inference_re_lu_2_layer_call_fn_45131inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_re_lu_2_layer_call_and_return_conditional_losses_45136inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBр
/__inference_max_pooling2d_2_layer_call_fn_45141inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_45146inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
%__inference_conv4_layer_call_fn_45155inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
@__inference_conv4_layer_call_and_return_conditional_losses_45165inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
Ю0
Я1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
·Bў
5__inference_batch_normalization_3_layer_call_fn_45178inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
5__inference_batch_normalization_3_layer_call_fn_45191inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_45209inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_45227inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
█B╪
'__inference_re_lu_3_layer_call_fn_45232inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_re_lu_3_layer_call_and_return_conditional_losses_45237inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBр
/__inference_max_pooling2d_3_layer_call_fn_45242inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_45247inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
%__inference_conv5_layer_call_fn_45256inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
@__inference_conv5_layer_call_and_return_conditional_losses_45266inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
╛0
┐1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
·Bў
5__inference_batch_normalization_4_layer_call_fn_45279inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
5__inference_batch_normalization_4_layer_call_fn_45292inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_45310inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_45328inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
█B╪
'__inference_re_lu_4_layer_call_fn_45333inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_re_lu_4_layer_call_and_return_conditional_losses_45338inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBр
/__inference_max_pooling2d_4_layer_call_fn_45343inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_45348inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
█B╪
'__inference_flatten_layer_call_fn_45353inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_flatten_layer_call_and_return_conditional_losses_45359inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
&__inference_dense1_layer_call_fn_45368inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_dense1_layer_call_and_return_conditional_losses_45379inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ьBщ
'__inference_dropout_layer_call_fn_45384inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ьBщ
'__inference_dropout_layer_call_fn_45389inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЗBД
B__inference_dropout_layer_call_and_return_conditional_losses_45394inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЗBД
B__inference_dropout_layer_call_and_return_conditional_losses_45406inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
%__inference_dense_layer_call_fn_45415inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
@__inference_dense_layer_call_and_return_conditional_losses_45426inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
Л	variables
М	keras_api

Нtotal

Оcount"
_tf_keras_metric
c
П	variables
Р	keras_api

Сtotal

Тcount
У
_fn_kwargs"
_tf_keras_metric
+:) 2Adam/m/conv1/kernel
+:) 2Adam/v/conv1/kernel
: 2Adam/m/conv1/bias
: 2Adam/v/conv1/bias
,:* 2 Adam/m/batch_normalization/gamma
,:* 2 Adam/v/batch_normalization/gamma
+:) 2Adam/m/batch_normalization/beta
+:) 2Adam/v/batch_normalization/beta
+:)  2Adam/m/conv2/kernel
+:)  2Adam/v/conv2/kernel
: 2Adam/m/conv2/bias
: 2Adam/v/conv2/bias
.:, 2"Adam/m/batch_normalization_1/gamma
.:, 2"Adam/v/batch_normalization_1/gamma
-:+ 2!Adam/m/batch_normalization_1/beta
-:+ 2!Adam/v/batch_normalization_1/beta
+:) @2Adam/m/conv3/kernel
+:) @2Adam/v/conv3/kernel
:@2Adam/m/conv3/bias
:@2Adam/v/conv3/bias
.:,@2"Adam/m/batch_normalization_2/gamma
.:,@2"Adam/v/batch_normalization_2/gamma
-:+@2!Adam/m/batch_normalization_2/beta
-:+@2!Adam/v/batch_normalization_2/beta
+:)@@2Adam/m/conv4/kernel
+:)@@2Adam/v/conv4/kernel
:@2Adam/m/conv4/bias
:@2Adam/v/conv4/bias
.:,@2"Adam/m/batch_normalization_3/gamma
.:,@2"Adam/v/batch_normalization_3/gamma
-:+@2!Adam/m/batch_normalization_3/beta
-:+@2!Adam/v/batch_normalization_3/beta
,:*@А2Adam/m/conv5/kernel
,:*@А2Adam/v/conv5/kernel
:А2Adam/m/conv5/bias
:А2Adam/v/conv5/bias
/:-А2"Adam/m/batch_normalization_4/gamma
/:-А2"Adam/v/batch_normalization_4/gamma
.:,А2!Adam/m/batch_normalization_4/beta
.:,А2!Adam/v/batch_normalization_4/beta
&:$
АА2Adam/m/dense1/kernel
&:$
АА2Adam/v/dense1/kernel
:А2Adam/m/dense1/bias
:А2Adam/v/dense1/bias
%:#
А╞
2Adam/m/dense/kernel
%:#
А╞
2Adam/v/dense/kernel
:╞
2Adam/m/dense/bias
:╞
2Adam/v/dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
&__inference_resize_layer_call_fn_45431inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_resize_layer_call_and_return_conditional_losses_45437inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
█B╪
'__inference_rescale_layer_call_fn_45442inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_rescale_layer_call_and_return_conditional_losses_45450inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
Н0
О1"
trackable_list_wrapper
.
Л	variables"
_generic_user_object
:  (2total
:  (2count
0
С0
Т1"
trackable_list_wrapper
.
П	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper╬
 __inference__wrapped_model_42922й223<=>?RS\]^_rs|}~ТУЬЭЮЯ▓│╝╜╛┐╪┘чшCв@
9в6
4К1
sequential_input         ЦЦ
к ".к+
)
dense К
dense         ╞
Є
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_45007Э\]^_MвJ
Cв@
:К7
inputs+                            
p 
к "FвC
<К9
tensor_0+                            
Ъ Є
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_45025Э\]^_MвJ
Cв@
:К7
inputs+                            
p
к "FвC
<К9
tensor_0+                            
Ъ ╠
5__inference_batch_normalization_1_layer_call_fn_44976Т\]^_MвJ
Cв@
:К7
inputs+                            
p 
к ";К8
unknown+                            ╠
5__inference_batch_normalization_1_layer_call_fn_44989Т\]^_MвJ
Cв@
:К7
inputs+                            
p
к ";К8
unknown+                            Є
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_45108Э|}~MвJ
Cв@
:К7
inputs+                           @
p 
к "FвC
<К9
tensor_0+                           @
Ъ Є
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_45126Э|}~MвJ
Cв@
:К7
inputs+                           @
p
к "FвC
<К9
tensor_0+                           @
Ъ ╠
5__inference_batch_normalization_2_layer_call_fn_45077Т|}~MвJ
Cв@
:К7
inputs+                           @
p 
к ";К8
unknown+                           @╠
5__inference_batch_normalization_2_layer_call_fn_45090Т|}~MвJ
Cв@
:К7
inputs+                           @
p
к ";К8
unknown+                           @Ў
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_45209бЬЭЮЯMвJ
Cв@
:К7
inputs+                           @
p 
к "FвC
<К9
tensor_0+                           @
Ъ Ў
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_45227бЬЭЮЯMвJ
Cв@
:К7
inputs+                           @
p
к "FвC
<К9
tensor_0+                           @
Ъ ╨
5__inference_batch_normalization_3_layer_call_fn_45178ЦЬЭЮЯMвJ
Cв@
:К7
inputs+                           @
p 
к ";К8
unknown+                           @╨
5__inference_batch_normalization_3_layer_call_fn_45191ЦЬЭЮЯMвJ
Cв@
:К7
inputs+                           @
p
к ";К8
unknown+                           @°
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_45310г╝╜╛┐NвK
DвA
;К8
inputs,                           А
p 
к "GвD
=К:
tensor_0,                           А
Ъ °
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_45328г╝╜╛┐NвK
DвA
;К8
inputs,                           А
p
к "GвD
=К:
tensor_0,                           А
Ъ ╥
5__inference_batch_normalization_4_layer_call_fn_45279Ш╝╜╛┐NвK
DвA
;К8
inputs,                           А
p 
к "<К9
unknown,                           А╥
5__inference_batch_normalization_4_layer_call_fn_45292Ш╝╜╛┐NвK
DвA
;К8
inputs,                           А
p
к "<К9
unknown,                           АЁ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_44906Э<=>?MвJ
Cв@
:К7
inputs+                            
p 
к "FвC
<К9
tensor_0+                            
Ъ Ё
N__inference_batch_normalization_layer_call_and_return_conditional_losses_44924Э<=>?MвJ
Cв@
:К7
inputs+                            
p
к "FвC
<К9
tensor_0+                            
Ъ ╩
3__inference_batch_normalization_layer_call_fn_44875Т<=>?MвJ
Cв@
:К7
inputs+                            
p 
к ";К8
unknown+                            ╩
3__inference_batch_normalization_layer_call_fn_44888Т<=>?MвJ
Cв@
:К7
inputs+                            
p
к ";К8
unknown+                            ╗
@__inference_conv1_layer_call_and_return_conditional_losses_44862w239в6
/в,
*К'
inputs         ЦЦ
к "6в3
,К)
tensor_0         ФФ 
Ъ Х
%__inference_conv1_layer_call_fn_44852l239в6
/в,
*К'
inputs         ЦЦ
к "+К(
unknown         ФФ ╖
@__inference_conv2_layer_call_and_return_conditional_losses_44963sRS7в4
-в*
(К%
inputs         JJ 
к "4в1
*К'
tensor_0         HH 
Ъ С
%__inference_conv2_layer_call_fn_44953hRS7в4
-в*
(К%
inputs         JJ 
к ")К&
unknown         HH ╖
@__inference_conv3_layer_call_and_return_conditional_losses_45064srs7в4
-в*
(К%
inputs         $$ 
к "4в1
*К'
tensor_0         ""@
Ъ С
%__inference_conv3_layer_call_fn_45054hrs7в4
-в*
(К%
inputs         $$ 
к ")К&
unknown         ""@╣
@__inference_conv4_layer_call_and_return_conditional_losses_45165uТУ7в4
-в*
(К%
inputs         @
к "4в1
*К'
tensor_0         @
Ъ У
%__inference_conv4_layer_call_fn_45155jТУ7в4
-в*
(К%
inputs         @
к ")К&
unknown         @║
@__inference_conv5_layer_call_and_return_conditional_losses_45266v▓│7в4
-в*
(К%
inputs         @
к "5в2
+К(
tensor_0         А
Ъ Ф
%__inference_conv5_layer_call_fn_45256k▓│7в4
-в*
(К%
inputs         @
к "*К'
unknown         Ам
A__inference_dense1_layer_call_and_return_conditional_losses_45379g╪┘0в-
&в#
!К
inputs         А
к "-в*
#К 
tensor_0         А
Ъ Ж
&__inference_dense1_layer_call_fn_45368\╪┘0в-
&в#
!К
inputs         А
к ""К
unknown         Ал
@__inference_dense_layer_call_and_return_conditional_losses_45426gчш0в-
&в#
!К
inputs         А
к "-в*
#К 
tensor_0         ╞

Ъ Е
%__inference_dense_layer_call_fn_45415\чш0в-
&в#
!К
inputs         А
к ""К
unknown         ╞
л
B__inference_dropout_layer_call_and_return_conditional_losses_45394e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ л
B__inference_dropout_layer_call_and_return_conditional_losses_45406e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ Е
'__inference_dropout_layer_call_fn_45384Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         АЕ
'__inference_dropout_layer_call_fn_45389Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         Ап
B__inference_flatten_layer_call_and_return_conditional_losses_45359i8в5
.в+
)К&
inputs         А
к "-в*
#К 
tensor_0         А
Ъ Й
'__inference_flatten_layer_call_fn_45353^8в5
.в+
)К&
inputs         А
к ""К
unknown         АЇ
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_45045еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╬
/__inference_max_pooling2d_1_layer_call_fn_45040ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    Ї
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_45146еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╬
/__inference_max_pooling2d_2_layer_call_fn_45141ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    Ї
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_45247еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╬
/__inference_max_pooling2d_3_layer_call_fn_45242ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    Ї
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_45348еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╬
/__inference_max_pooling2d_4_layer_call_fn_45343ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    Є
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_44944еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╠
-__inference_max_pooling2d_layer_call_fn_44939ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ╡
B__inference_re_lu_1_layer_call_and_return_conditional_losses_45035o7в4
-в*
(К%
inputs         HH 
к "4в1
*К'
tensor_0         HH 
Ъ П
'__inference_re_lu_1_layer_call_fn_45030d7в4
-в*
(К%
inputs         HH 
к ")К&
unknown         HH ╡
B__inference_re_lu_2_layer_call_and_return_conditional_losses_45136o7в4
-в*
(К%
inputs         ""@
к "4в1
*К'
tensor_0         ""@
Ъ П
'__inference_re_lu_2_layer_call_fn_45131d7в4
-в*
(К%
inputs         ""@
к ")К&
unknown         ""@╡
B__inference_re_lu_3_layer_call_and_return_conditional_losses_45237o7в4
-в*
(К%
inputs         @
к "4в1
*К'
tensor_0         @
Ъ П
'__inference_re_lu_3_layer_call_fn_45232d7в4
-в*
(К%
inputs         @
к ")К&
unknown         @╖
B__inference_re_lu_4_layer_call_and_return_conditional_losses_45338q8в5
.в+
)К&
inputs         А
к "5в2
+К(
tensor_0         А
Ъ С
'__inference_re_lu_4_layer_call_fn_45333f8в5
.в+
)К&
inputs         А
к "*К'
unknown         А╖
@__inference_re_lu_layer_call_and_return_conditional_losses_44934s9в6
/в,
*К'
inputs         ФФ 
к "6в3
,К)
tensor_0         ФФ 
Ъ С
%__inference_re_lu_layer_call_fn_44929h9в6
/в,
*К'
inputs         ФФ 
к "+К(
unknown         ФФ ╣
B__inference_rescale_layer_call_and_return_conditional_losses_45450s9в6
/в,
*К'
inputs         ЦЦ
к "6в3
,К)
tensor_0         ЦЦ
Ъ У
'__inference_rescale_layer_call_fn_45442h9в6
/в,
*К'
inputs         ЦЦ
к "+К(
unknown         ЦЦ╕
A__inference_resize_layer_call_and_return_conditional_losses_45437s9в6
/в,
*К'
inputs         ЦЦ
к "6в3
,К)
tensor_0         ЦЦ
Ъ Т
&__inference_resize_layer_call_fn_45431h9в6
/в,
*К'
inputs         ЦЦ
к "+К(
unknown         ЦЦ№
G__inference_sequential_1_layer_call_and_return_conditional_losses_44212░223<=>?RS\]^_rs|}~ТУЬЭЮЯ▓│╝╜╛┐╪┘чшKвH
Aв>
4К1
sequential_input         ЦЦ
p 

 
к "-в*
#К 
tensor_0         ╞

Ъ №
G__inference_sequential_1_layer_call_and_return_conditional_losses_44309░223<=>?RS\]^_rs|}~ТУЬЭЮЯ▓│╝╜╛┐╪┘чшKвH
Aв>
4К1
sequential_input         ЦЦ
p

 
к "-в*
#К 
tensor_0         ╞

Ъ Є
G__inference_sequential_1_layer_call_and_return_conditional_losses_44669ж223<=>?RS\]^_rs|}~ТУЬЭЮЯ▓│╝╜╛┐╪┘чшAв>
7в4
*К'
inputs         ЦЦ
p 

 
к "-в*
#К 
tensor_0         ╞

Ъ Є
G__inference_sequential_1_layer_call_and_return_conditional_losses_44813ж223<=>?RS\]^_rs|}~ТУЬЭЮЯ▓│╝╜╛┐╪┘чшAв>
7в4
*К'
inputs         ЦЦ
p

 
к "-в*
#К 
tensor_0         ╞

Ъ ╓
,__inference_sequential_1_layer_call_fn_43670е223<=>?RS\]^_rs|}~ТУЬЭЮЯ▓│╝╜╛┐╪┘чшKвH
Aв>
4К1
sequential_input         ЦЦ
p 

 
к ""К
unknown         ╞
╓
,__inference_sequential_1_layer_call_fn_44115е223<=>?RS\]^_rs|}~ТУЬЭЮЯ▓│╝╜╛┐╪┘чшKвH
Aв>
4К1
sequential_input         ЦЦ
p

 
к ""К
unknown         ╞
╠
,__inference_sequential_1_layer_call_fn_44459Ы223<=>?RS\]^_rs|}~ТУЬЭЮЯ▓│╝╜╛┐╪┘чшAв>
7в4
*К'
inputs         ЦЦ
p 

 
к ""К
unknown         ╞
╠
,__inference_sequential_1_layer_call_fn_44532Ы223<=>?RS\]^_rs|}~ТУЬЭЮЯ▓│╝╜╛┐╪┘чшAв>
7в4
*К'
inputs         ЦЦ
p

 
к ""К
unknown         ╞
╦
E__inference_sequential_layer_call_and_return_conditional_losses_42990БGвD
=в:
0К-
resize_input         ЦЦ
p 

 
к "6в3
,К)
tensor_0         ЦЦ
Ъ ╦
E__inference_sequential_layer_call_and_return_conditional_losses_42996БGвD
=в:
0К-
resize_input         ЦЦ
p

 
к "6в3
,К)
tensor_0         ЦЦ
Ъ ─
E__inference_sequential_layer_call_and_return_conditional_losses_44833{Aв>
7в4
*К'
inputs         ЦЦ
p 

 
к "6в3
,К)
tensor_0         ЦЦ
Ъ ─
E__inference_sequential_layer_call_and_return_conditional_losses_44843{Aв>
7в4
*К'
inputs         ЦЦ
p

 
к "6в3
,К)
tensor_0         ЦЦ
Ъ д
*__inference_sequential_layer_call_fn_42951vGвD
=в:
0К-
resize_input         ЦЦ
p 

 
к "+К(
unknown         ЦЦд
*__inference_sequential_layer_call_fn_42984vGвD
=в:
0К-
resize_input         ЦЦ
p

 
к "+К(
unknown         ЦЦЮ
*__inference_sequential_layer_call_fn_44818pAв>
7в4
*К'
inputs         ЦЦ
p 

 
к "+К(
unknown         ЦЦЮ
*__inference_sequential_layer_call_fn_44823pAв>
7в4
*К'
inputs         ЦЦ
p

 
к "+К(
unknown         ЦЦх
#__inference_signature_wrapper_44386╜223<=>?RS\]^_rs|}~ТУЬЭЮЯ▓│╝╜╛┐╪┘чшWвT
в 
MкJ
H
sequential_input4К1
sequential_input         ЦЦ".к+
)
dense К
dense         ╞
