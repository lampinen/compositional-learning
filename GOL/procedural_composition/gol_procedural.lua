require "torch"
require "nn"
---------------------------------------------
--Using neural networks to learn the rules
--of conway's game of life
--------------------------------------------


--learning parameters
num_training_points = 256
num_iterations = 50
num_averaging_trials = 100
initial_learning_rate = 0.01
learning_rate_decay_multiplier = 0.99
output_figure_name = '-n-'.. num_training_points .. '-AVG-lrate-' .. initial_learning_rate ..'-ldecay-' .. learning_rate_decay_multiplier .. '.png'

--Helper functions------------------------------------

function random_pm1_T(n) --Returns a random Tensor with n elements all +/- 1
	return torch.floor(torch.rand(n)+0.5)*2-1
end

function threshold(T) --Returns Tensor thresholded at 0 to +/- 1
	local new_T = T:clone():zero()-1
	new_T[T:gt(0)] = 1
	return new_T
end

function three_square_pm_T(i) -- Returns the ith tensor from the enumeration of all 512 tensors with 9 elements +/- 1
	new_T = torch.Tensor(9):zero() -1
	j = 0
	new_T:apply(function() if math.floor((i-1)/(2^j)) % 2 == 1 then j = j + 1;return 1; else j = j + 1;return 0; end;  end) 
	return (new_T:reshape(3,3))*2-1
end

compositional_finals_track = torch.Tensor(num_averaging_trials+1):zero()

compositional_avg_track = torch.Tensor(num_iterations+1):zero()
compositional_sloppy_avg_track = torch.Tensor(num_iterations+1):zero()
standard_avg_track = torch.Tensor(num_iterations+1):zero()
standard_ReLU_avg_track = torch.Tensor(num_iterations+1):zero()
standard_3_avg_track = torch.Tensor(num_iterations+1):zero()
compositional_MSE_avg_track = torch.Tensor(num_iterations+1):zero()
compositional_sloppy_MSE_avg_track = torch.Tensor(num_iterations+1):zero()
standard_MSE_avg_track = torch.Tensor(num_iterations+1):zero()
standard_ReLU_MSE_avg_track = torch.Tensor(num_iterations+1):zero()
standard_3_MSE_avg_track = torch.Tensor(num_iterations+1):zero()
for rseed = 1,num_averaging_trials do
	print("On averaging step: " .. rseed)
	torch.manualSeed(rseed) --New run! 
	learning_rate = initial_learning_rate
	--build training data----------------------------------------------


	trainset = {};
	trainset_number = {};
	trainset_select = {};
	compositional_trainset = {};
	testset = {};
	testset_number = {};

	data_order = torch.randperm(512)
	--Divide data into training, testing 
	local num_test_points = 512-num_training_points
	for i=1,num_training_points do 
		local input = three_square_pm_T(data_order[i])
		local output = torch.Tensor(1);
		local output_number = torch.Tensor(1);
		local output_select = torch.Tensor(1);
		output_number[1] = torch.sum(input)-input[2][2]
		output_select[1] = input[2][2] 
		if output_number[1] > 3 or output_number[1] < 2 then
			output[1] = -1
		elseif output_number[1] == 3 then
			output[1] = 1
		else
			output[1] = input[2][2]	
		end
		input = input:reshape(9)
		trainset[i] = {input, output}
		trainset_number[i] = {input, output_number}
		trainset_select[i] = {input, output_select}
		compositional_trainset[i] = {torch.Tensor({output_number[1],input[5]}),output}

	end
	function trainset:size() return table.getn(trainset) end 
	function trainset_number:size() return table.getn(trainset_number) end 
	function trainset_select:size() return table.getn(trainset_select) end 
	function compositional_trainset:size() return table.getn(compositional_trainset) end 


	for i=1,num_test_points do 
		local input = three_square_pm_T(data_order[i+num_training_points])
		local output = torch.Tensor(1);
		local output_number = torch.Tensor(1);
		output_number[1] = torch.sum(input)-input[2][2]
		if output_number[1] > 3 or output_number[1] < 2 then
			output[1] = -1
		elseif output_number[1] == 3 then
			output[1] = 1
		else
			output[1] = input[2][2]	
		end
		input = input:reshape(9)
		testset[i] = {input, output}
		testset_number[i] = {input, output_number}
	end
	function testset:size() return table.getn(testset) end 
	function testset_number:size() return table.getn(testset_number) end 

	--Network training----------------------------------------------------------

	inputs = 9
	compositional_inputs = 2
	HUs = 10
	outputs = 1


	--Standard net-------------------------------------------------------
	standard_net = nn.Sequential();
	standard_net:add(nn.Linear(inputs, HUs))
	standard_net:add(nn.Tanh())
	standard_net:add(nn.Linear(HUs, outputs))
	standard_net:add(nn.Tanh())

	--Standard ReLU first layer---------------------------
	standard_ReLU_net = nn.Sequential();
	standard_ReLU_net:add(nn.Linear(inputs, HUs))
	standard_ReLU_net:add(nn.ReLU())
	standard_ReLU_net:add(nn.Linear(HUs, outputs))
	standard_ReLU_net:add(nn.Tanh())

	--Standard 3-layer ReLU first layer -----------------------------
	standard_3_net = nn.Sequential();
	standard_3_net:add(nn.Linear(inputs, HUs))
	standard_3_net:add(nn.ReLU())
	standard_3_net:add(nn.Linear(HUs, HUs))
	standard_3_net:add(nn.Tanh())
	standard_3_net:add(nn.Linear(HUs, outputs))
	standard_3_net:add(nn.Tanh())

	--compositional net-----------------------------------------------
	--Count net
	count_net = nn.Sequential();
	count_net:add(nn.Linear(inputs, outputs))
	count_net:add(nn.ReLU())

	--select net
	select_net = nn.Sequential();
	select_net:add(nn.Linear(inputs, outputs))
	select_net:add(nn.ReLU())

	--compositional_net

	compositional_net = nn.Sequential();
	compositional_net:add(nn.Linear(compositional_inputs, HUs))
	compositional_net:add(nn.Tanh())
	compositional_net:add(nn.Linear(HUs, outputs))
	compositional_net:add(nn.Tanh())

	--Alternate: instead of training with true values, trains with outputs of the count_net
	compositional_sloppy_net = nn.Sequential();
	compositional_sloppy_net:add(nn.Linear(compositional_inputs, HUs))
	compositional_sloppy_net:add(nn.Tanh())
	compositional_sloppy_net:add(nn.Linear(HUs, outputs))
	compositional_sloppy_net:add(nn.Tanh())

	--Manual training, for plotting
	compositional_track = {}
	compositional_sloppy_track = {}
	standard_track = {}
	standard_ReLU_track = {}
	standard_3_track = {}
	compositional_MSE_track = {}
	compositional_sloppy_MSE_track = {}
	standard_MSE_track = {}
	standard_ReLU_MSE_track = {}
	standard_3_MSE_track = {}
	--pre-training results
	s1_thresh_error = 0
	s2_thresh_error = 0
	s3_thresh_error = 0
	h_thresh_error = 0
	hs_thresh_error = 0
	for i = 1,num_test_points do
		s1_thresh_error = s1_thresh_error + torch.abs(threshold(standard_net:forward(testset[i][1]))[1]-testset[i][2][1])/2
		s2_thresh_error = s2_thresh_error + torch.abs(threshold(standard_ReLU_net:forward(testset[i][1]))[1]-testset[i][2][1])/2
		s3_thresh_error = s3_thresh_error + torch.abs(threshold(standard_3_net:forward(testset[i][1]))[1]-testset[i][2][1])/2
		local compositional_input = torch.Tensor({count_net:forward(testset[i][1])[1],select_net:forward(testset[i][1])[1]})
		h_thresh_error = h_thresh_error + torch.abs(threshold(compositional_net:forward(compositional_input))[1]-testset[i][2][1])/2
		hs_thresh_error = hs_thresh_error + torch.abs(threshold(compositional_sloppy_net:forward(compositional_input))[1]-testset[i][2][1])/2
	end
	s1_thresh_error = s1_thresh_error/num_test_points
	s2_thresh_error = s2_thresh_error/num_test_points
	s3_thresh_error = s3_thresh_error/num_test_points
	h_thresh_error = h_thresh_error/num_test_points
	hs_thresh_error = hs_thresh_error/num_test_points
	standard_track[1] = s1_thresh_error
	standard_ReLU_track[1] = s2_thresh_error
	standard_3_track[1] = s3_thresh_error
	compositional_track[1] = h_thresh_error
	compositional_sloppy_track[1] = hs_thresh_error

	s1_MS_error = 0
	s2_MS_error = 0
	s3_MS_error = 0
	h_MS_error = 0
	hs_MS_error = 0
	for i = 1,num_test_points do
		s1_MS_error = s1_MS_error + ((standard_net:forward(testset[i][1]))[1]-testset[i][2][1])^2
		s2_MS_error = s2_MS_error + ((standard_ReLU_net:forward(testset[i][1]))[1]-testset[i][2][1])^2
		s3_MS_error = s3_MS_error + ((standard_3_net:forward(testset[i][1]))[1]-testset[i][2][1])^2
		local compositional_input =  torch.Tensor({count_net:forward(testset[i][1])[1],select_net:forward(testset[i][1])[1]})
		h_MS_error = h_MS_error + ((compositional_net:forward(compositional_input))[1]-testset[i][2][1])^2
		hs_MS_error = hs_MS_error + ((compositional_sloppy_net:forward(compositional_input))[1]-testset[i][2][1])^2
	end
	s1_MS_error = s1_MS_error/num_test_points
	s2_MS_error = s2_MS_error/num_test_points
	s3_MS_error = s3_MS_error/num_test_points
	h_MS_error = h_MS_error/num_test_points
	hs_MS_error = hs_MS_error/num_test_points
	standard_MSE_track[1] = s1_MS_error
	standard_ReLU_MSE_track[1] = s2_MS_error
	standard_3_MSE_track[1] = s3_MS_error
	compositional_MSE_track[1] = h_MS_error
	compositional_sloppy_MSE_track[1] = hs_MS_error


	for iteration = 1,num_iterations do
		training_order = torch.randperm(num_training_points)
		learning_rate = learning_rate * learning_rate_decay_multiplier
		for point_i = 1,num_training_points do
			i = training_order[point_i] 
			criterion = nn.MSECriterion()  
			--standard
			criterion:forward(standard_net:forward(trainset[i][1]),trainset[i][2])	
			standard_net:zeroGradParameters()		
			standard_net:backward(trainset[i][1],criterion:backward(standard_net.output,trainset[i][2]))
			standard_net:updateParameters(learning_rate)
			--standard_ReLU
			criterion:forward(standard_ReLU_net:forward(trainset[i][1]),trainset[i][2])	
			standard_ReLU_net:zeroGradParameters()		
			standard_ReLU_net:backward(trainset[i][1],criterion:backward(standard_ReLU_net.output,trainset[i][2]))
			standard_ReLU_net:updateParameters(learning_rate)
			--standard_3
			criterion:forward(standard_3_net:forward(trainset[i][1]),trainset[i][2])	
			standard_3_net:zeroGradParameters()		
			standard_3_net:backward(trainset[i][1],criterion:backward(standard_3_net.output,trainset[i][2]))
			standard_3_net:updateParameters(learning_rate)
			--count
			criterion:forward(count_net:forward(trainset_number[i][1]),trainset_number[i][2])	
			count_net:zeroGradParameters()		
			count_net:backward(trainset_number[i][1],criterion:backward(count_net.output,trainset_number[i][2]))
			count_net:updateParameters(learning_rate)
			--select
			criterion:forward(select_net:forward(trainset_select[i][1]),trainset_select[i][2])	
			select_net:zeroGradParameters()		
			select_net:backward(trainset_select[i][1],criterion:backward(select_net.output,trainset_select[i][2]))
			select_net:updateParameters(learning_rate)
			--compositional
			local compositional_input = compositional_trainset[i][1]
			criterion:forward(compositional_net:forward(compositional_input),trainset[i][2])	
			compositional_net:zeroGradParameters()		
			compositional_net:backward(compositional_input,criterion:backward(compositional_net.output,trainset[i][2]))
			compositional_net:updateParameters(learning_rate)
			--compositional sloppy
			local compositional_input = torch.Tensor({count_net:forward(trainset[i][1])[1],select_net:forward(trainset[i][1])[1]})
			criterion:forward(compositional_sloppy_net:forward(compositional_input),trainset[i][2])	
			compositional_sloppy_net:zeroGradParameters()		
			compositional_sloppy_net:backward(compositional_input,criterion:backward(compositional_sloppy_net.output,trainset[i][2]))
			compositional_sloppy_net:updateParameters(learning_rate)
		end

		--Results on this iteration
		s1_thresh_error = 0
		s2_thresh_error = 0
		s3_thresh_error = 0
		h_thresh_error = 0
		hs_thresh_error = 0
		for i = 1,num_test_points do
			s1_thresh_error = s1_thresh_error + torch.abs(threshold(standard_net:forward(testset[i][1]))[1]-testset[i][2][1])/2
			s2_thresh_error = s2_thresh_error + torch.abs(threshold(standard_ReLU_net:forward(testset[i][1]))[1]-testset[i][2][1])/2
			s3_thresh_error = s3_thresh_error + torch.abs(threshold(standard_3_net:forward(testset[i][1]))[1]-testset[i][2][1])/2
			local compositional_input =  torch.Tensor({count_net:forward(testset[i][1])[1],select_net:forward(testset[i][1])[1]})
			h_thresh_error = h_thresh_error + torch.abs(threshold(compositional_net:forward(compositional_input))[1]-testset[i][2][1])/2
			hs_thresh_error = hs_thresh_error + torch.abs(threshold(compositional_sloppy_net:forward(compositional_input))[1]-testset[i][2][1])/2
		end
		s1_thresh_error = s1_thresh_error/num_test_points
		s2_thresh_error = s2_thresh_error/num_test_points
		s3_thresh_error = s3_thresh_error/num_test_points
		h_thresh_error = h_thresh_error/num_test_points
		hs_thresh_error = hs_thresh_error/num_test_points
		standard_track[iteration+1] = s1_thresh_error
		standard_ReLU_track[iteration+1] = s2_thresh_error
		standard_3_track[iteration+1] = s3_thresh_error
		compositional_track[iteration+1] = h_thresh_error
		compositional_sloppy_track[iteration+1] = hs_thresh_error

		s1_MS_error = 0
		s2_MS_error = 0
		s3_MS_error = 0
		h_MS_error = 0
		hs_MS_error = 0
		for i = 1,num_test_points do
			s1_MS_error = s1_MS_error + ((standard_net:forward(testset[i][1]))[1]-testset[i][2][1])^2
			s2_MS_error = s2_MS_error + ((standard_ReLU_net:forward(testset[i][1]))[1]-testset[i][2][1])^2
			s3_MS_error = s3_MS_error + ((standard_3_net:forward(testset[i][1]))[1]-testset[i][2][1])^2
			local compositional_input = torch.Tensor({count_net:forward(testset[i][1])[1],select_net:forward(testset[i][1])[1]})
			h_MS_error = h_MS_error + ((compositional_net:forward(compositional_input))[1]-testset[i][2][1])^2
			hs_MS_error = hs_MS_error + ((compositional_sloppy_net:forward(compositional_input))[1]-testset[i][2][1])^2
		end
		s1_MS_error = s1_MS_error/num_test_points
		s2_MS_error = s2_MS_error/num_test_points
		s3_MS_error = s3_MS_error/num_test_points
		h_MS_error = h_MS_error/num_test_points
		hs_MS_error = hs_MS_error/num_test_points
		standard_MSE_track[iteration+1] = s1_MS_error
		standard_ReLU_MSE_track[iteration+1] = s2_MS_error
		standard_3_MSE_track[iteration+1] = s3_MS_error
		compositional_MSE_track[iteration+1] = h_MS_error
		compositional_sloppy_MSE_track[iteration+1] = hs_MS_error
	end
	compositional_avg_track = compositional_avg_track+torch.Tensor(compositional_track)
	compositional_sloppy_avg_track = compositional_sloppy_avg_track + torch.Tensor(compositional_sloppy_track)
	standard_avg_track = standard_avg_track + torch.Tensor(standard_track)
	standard_ReLU_avg_track = standard_ReLU_avg_track + torch.Tensor(standard_ReLU_track)
	standard_3_avg_track = standard_3_avg_track + torch.Tensor(standard_3_track)
	compositional_MSE_avg_track = compositional_MSE_avg_track + torch.Tensor(compositional_MSE_track)
	compositional_sloppy_MSE_avg_track = compositional_sloppy_MSE_avg_track + torch.Tensor(compositional_sloppy_MSE_track)
	standard_MSE_avg_track = standard_MSE_avg_track + torch.Tensor(standard_MSE_track)
	standard_ReLU_MSE_avg_track = standard_ReLU_MSE_avg_track + torch.Tensor(standard_ReLU_MSE_track)
	standard_3_MSE_avg_track = standard_3_MSE_avg_track + torch.Tensor(standard_3_MSE_track)
	compositional_finals_track[rseed] = compositional_sloppy_track[num_iterations+1]
	print(compositional_finals_track)
end

compositional_avg_track = compositional_avg_track / num_averaging_trials 
compositional_sloppy_avg_track = compositional_sloppy_avg_track  / num_averaging_trials
standard_avg_track = standard_avg_track / num_averaging_trials
standard_ReLU_avg_track = standard_ReLU_avg_track / num_averaging_trials
standard_3_avg_track = standard_3_avg_track / num_averaging_trials
compositional_MSE_avg_track = compositional_MSE_avg_track / num_averaging_trials
compositional_sloppy_MSE_avg_track = compositional_sloppy_MSE_avg_track / num_averaging_trials
standard_MSE_avg_track = standard_MSE_avg_track / num_averaging_trials
standard_ReLU_MSE_avg_track = standard_ReLU_MSE_avg_track / num_averaging_trials
standard_3_MSE_avg_track = standard_3_MSE_avg_track / num_averaging_trials
----Plot results---------------------------------------------

require 'gnuplot'

iterations = torch.linspace(0,num_iterations,num_iterations+1)



gnuplot.pngfigure('error-rate-plot'..output_figure_name)
--gnuplot.raw('set size square 0.9,0.9') --For mini abstract figures
gnuplot.plot({'standard (1 HL, Tanh)',iterations,standard_avg_track},{'standard (1 HL, ReLU)',iterations,standard_ReLU_avg_track},{'standard (2 HL, ReLU, Tanh)',iterations,standard_3_avg_track},{'compositional',iterations,compositional_sloppy_avg_track})
gnuplot.title('Incorrect response rate after thresholding')
gnuplot.xlabel('Epoch')
gnuplot.ylabel('Test error rate')
gnuplot.pngfigure('MSE-plot'..output_figure_name)
--gnuplot.raw('set size square 0.9,0.9')
gnuplot.plot({'standard (1 HL, Tanh)',iterations,standard_MSE_avg_track},{'standard (1 HL, ReLU)',iterations,standard_ReLU_MSE_avg_track},{'standard (2 HL, ReLU, Tanh)',iterations,standard_3_MSE_avg_track},{'compositional',iterations,compositional_sloppy_MSE_avg_track})
gnuplot.title('Response MSE, w/o thresholding')
gnuplot.xlabel('Epoch')
gnuplot.ylabel('Test MSE')
