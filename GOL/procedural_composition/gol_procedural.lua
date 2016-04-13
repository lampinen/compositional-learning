require "torch"
require "nn"
require "rnn"
---------------------------------------------
--Using neural networks to learn the rules
--of conway's game of life, by procedural
--composition (i.e., by teaching a network
--to use a set of procedures in a sequence
--rather than building the sequence explicitly
--into the network)
--------------------------------------------


--learning parameters---------------------------
--data/time parameters
num_averaging_trials = 100
num_iterations = 50
num_training_points = 64

--feedforward parameters
initial_learning_rate = 0.01
learning_rate_decay_multiplier = 0.99

grid_inputs = 9
compositional_inputs = 2
HUs = 10
outputs = 1


--recurrent parameters
rho = 5 -- sequence length (time backprop cutoff)
codeSize = 10 -- Codes for the various procedures 
recurrent_learning_rate = 0.1

--miscellaneous
output_figure_name = '-n-'.. num_training_points .. '-fflrate-' .. initial_learning_rate .. '-rlrate' .. recurrent_learning_rate ..'-ldecay-' .. learning_rate_decay_multiplier .. '.png'

--Helper functions------------------------------------

function random_pm1_T(n) --Returns a random Tensor with n elements all +/- 1
	return torch.floor(torch.rand(n)+0.5)*2-1
end

function threshold(T) --Returns Tensor thresholded at 0 to +/- 1
	local new_T = T:clone():zero()-1
	new_T[T:gt(0)] = 1
	return new_T
end

function tensor_shallow_eq(S,T)
	return torch.all(torch.eq(S,T))
end

function three_square_pm_T(i) -- Returns the ith tensor from the enumeration of all 512 tensors with 9 elements +/- 1
	new_T = torch.Tensor(9):zero() -1
	j = 0
	new_T:apply(function() if math.floor((i-1)/(2^j)) % 2 == 1 then j = j + 1;return 1; else j = j + 1;return 0; end;  end) 
	return (new_T:reshape(3,3))*2-1
end

compositional_finals_track = torch.Tensor(num_averaging_trials+1):zero()

compositional_avg_track = torch.Tensor(num_iterations+1):zero()
standard_avg_track = torch.Tensor(num_iterations+1):zero()
standard_ReLU_avg_track = torch.Tensor(num_iterations+1):zero()
standard_3_avg_track = torch.Tensor(num_iterations+1):zero()
compositional_MSE_avg_track = torch.Tensor(num_iterations+1):zero()
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
	compositional_trainset = {};
	testset = {};

	data_order = torch.randperm(512)
	--Divide data into training, testing 
	local num_test_points = 512-num_training_points
	for i=1,num_training_points do 
		local input = three_square_pm_T(data_order[i])
		local output = torch.Tensor(1);
		local output_number = torch.Tensor(2);
		output_number[1] = torch.sum(input)-input[2][2]
		output_number[2] = input[2][2] 
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
		compositional_trainset[i] = {torch.Tensor({output_number[1],input[5]}),output}

	end
	function trainset:size() return table.getn(trainset) end 
	function trainset_number:size() return table.getn(trainset_number) end 
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
	end
	function testset:size() return table.getn(testset) end 
	
	
	--RNN data------------
	--codes for compositional
	GOL_procedure_code = torch.Tensor(codeSize):zero() -1
	GOL_procedure_code[1] = 1
	feature_extraction_code =  torch.Tensor(codeSize):zero() -1
	feature_extraction_code[2] = 1
	life_comp_code =  torch.Tensor(codeSize):zero() -1
	life_comp_code[3] = 1
	RETURN_code = torch.Tensor(codeSize):zero() -1

	--Create a sequence for this task 
	local GOL_procedure_length = 3
	local inputs, targets = {}, {}
	inputs[1] = GOL_procedure_code
	inputs[2] = torch.Tensor(codeSize):zero()
	inputs[3] = torch.Tensor(codeSize):zero()
	targets[1] = feature_extraction_code
	targets[2] = life_comp_code
	targets[3] = RETURN_code


	--Network training----------------------------------------------------------

	--Standard net-------------------------------------------------------
	standard_net = nn.Sequential();
	standard_net:add(nn.Linear(grid_inputs, HUs))
	standard_net:add(nn.Tanh())
	standard_net:add(nn.Linear(HUs, outputs))
	standard_net:add(nn.Tanh())

	--Standard ReLU first layer---------------------------
	standard_ReLU_net = nn.Sequential();
	standard_ReLU_net:add(nn.Linear(grid_inputs, HUs))
	standard_ReLU_net:add(nn.ReLU())
	standard_ReLU_net:add(nn.Linear(HUs, outputs))
	standard_ReLU_net:add(nn.Tanh())

	--Standard 3-layer ReLU first layer -----------------------------
	standard_3_net = nn.Sequential();
	standard_3_net:add(nn.Linear(grid_inputs, HUs))
	standard_3_net:add(nn.ReLU())
	standard_3_net:add(nn.Linear(HUs, HUs))
	standard_3_net:add(nn.Tanh())
	standard_3_net:add(nn.Linear(HUs, outputs))
	standard_3_net:add(nn.Tanh())

	--compositional net-----------------------------------------------
	--procedure net: chooses what to do next
	local r = nn.Recurrent(
	   HUs, nn.Linear(codeSize, HUs),
	   nn.Linear(HUs, HUs), nn.Tanh(),
	   rho
	)  
	   
	local rnn = nn.Sequential()
	   :add(r)
	   :add(nn.Linear(HUs, HUs))
	   :add(nn.Tanh())

	procedure_net = nn.Recursor(rnn, rho)

	--feature extraction net
	feature_net = nn.Sequential();
	feature_net:add(nn.Linear(grid_inputs, compositional_inputs))
	feature_net:add(nn.ReLU())

	--life_comp_net, step 2
	life_comp_net = nn.Sequential();
	life_comp_net:add(nn.Linear(compositional_inputs, HUs))
	life_comp_net:add(nn.Tanh())
	life_comp_net:add(nn.Linear(HUs, outputs))
	life_comp_net:add(nn.Tanh())

	--Evaluation function
	function compositional_net_forward(compositional_net_input,this_procedure_code) 
		procedure_net:forget()
		local procedure_next = threshold(procedure_net:forward(this_procedure_code))
		while (not tensor_shallow_eq(procedure_next,RETURN_code)) do 
			if tensor_shallow_eq(procedure_next,feature_extraction_code) then
				compositional_net_input = feature_net:forward(compositional_net_input)	
			elseif tensor_shallow_eq(procedure_next,life_comp_code) then
				compositional_net_input = life_comp_net:forward(compositional_net_input)	
			else --Unknown procedure, let's just skip	
				break	
			end
			procedure_next = threshold(procedure_net:forward(torch.Tensor(codeSize):zero()))
		end

		return compositional_net_input
	end

	--Training-------------------------------------------------------------
	compositional_track = {}
	standard_track = {}
	standard_ReLU_track = {}
	standard_3_track = {}
	compositional_MSE_track = {}
	standard_MSE_track = {}
	standard_ReLU_MSE_track = {}
	standard_3_MSE_track = {}
	--pre-training results
	s1_thresh_error = 0
	s2_thresh_error = 0
	s3_thresh_error = 0
	hs_thresh_error = 0
	for i = 1,num_test_points do
		s1_thresh_error = s1_thresh_error + torch.abs(threshold(standard_net:forward(testset[i][1]))[1]-testset[i][2][1])/2
		s2_thresh_error = s2_thresh_error + torch.abs(threshold(standard_ReLU_net:forward(testset[i][1]))[1]-testset[i][2][1])/2
		s3_thresh_error = s3_thresh_error + torch.abs(threshold(standard_3_net:forward(testset[i][1]))[1]-testset[i][2][1])/2
		
		--compositional
		local this_procedure_code = GOL_procedure_code
		local compositional_net_input = testset[i][1]
		local compositional_net_output = compositional_net_forward(compositional_net_input,this_procedure_code)
		hs_thresh_error = hs_thresh_error + torch.abs(threshold(compositional_net_output)[1]-testset[i][2][1])/2
	end
	s1_thresh_error = s1_thresh_error/num_test_points
	s2_thresh_error = s2_thresh_error/num_test_points
	s3_thresh_error = s3_thresh_error/num_test_points
	hs_thresh_error = hs_thresh_error/num_test_points
	standard_track[1] = s1_thresh_error
	standard_ReLU_track[1] = s2_thresh_error
	standard_3_track[1] = s3_thresh_error
	compositional_track[1] = hs_thresh_error

	s1_MS_error = 0
	s2_MS_error = 0
	s3_MS_error = 0
	hs_MS_error = 0
	for i = 1,num_test_points do
		s1_MS_error = s1_MS_error + ((standard_net:forward(testset[i][1]))[1]-testset[i][2][1])^2
		s2_MS_error = s2_MS_error + ((standard_ReLU_net:forward(testset[i][1]))[1]-testset[i][2][1])^2
		s3_MS_error = s3_MS_error + ((standard_3_net:forward(testset[i][1]))[1]-testset[i][2][1])^2
		--compositional
		local this_procedure_code = GOL_procedure_code
		local compositional_net_input = testset[i][1]
		local compositional_net_output = compositional_net_forward(compositional_net_input,this_procedure_code)
		hs_MS_error = hs_MS_error + ((compositional_net_output)[1]-testset[i][2][1])^2
	end
	s1_MS_error = s1_MS_error/num_test_points
	s2_MS_error = s2_MS_error/num_test_points
	s3_MS_error = s3_MS_error/num_test_points
	hs_MS_error = hs_MS_error/num_test_points
	standard_MSE_track[1] = s1_MS_error
	standard_ReLU_MSE_track[1] = s2_MS_error
	standard_3_MSE_track[1] = s3_MS_error
	compositional_MSE_track[1] = hs_MS_error


	for iteration = 1,num_iterations do
		training_order = torch.randperm(num_training_points)
		learning_rate = learning_rate * learning_rate_decay_multiplier
		recurrent_learning_rate = recurrent_learning_rate * learning_rate_decay_multiplier
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
			--feature
			criterion:forward(feature_net:forward(trainset_number[i][1]),trainset_number[i][2])	
			feature_net:zeroGradParameters()		
			feature_net:backward(trainset_number[i][1],criterion:backward(feature_net.output,trainset_number[i][2]))
			feature_net:updateParameters(learning_rate)
			--compositional
			local compositional_input = feature_net:forward(trainset[i][1])
			criterion:forward(life_comp_net:forward(compositional_input),trainset[i][2])	
			life_comp_net:zeroGradParameters()		
			life_comp_net:backward(compositional_input,criterion:backward(life_comp_net.output,trainset[i][2]))
			life_comp_net:updateParameters(learning_rate)
			--procedural
			procedure_net:zeroGradParameters()
			procedure_net:forget() -- forget all past time-steps

			local outputs, err = {}, 0
			for step=1,GOL_procedure_length do
				outputs[step] = procedure_net:forward(inputs[step])
				err = err + criterion:forward(outputs[step], targets[step])
			end

			--backward sequence through procedure_net (i.e. backprop through time)

			local gradOutputs, gradInputs = {}, {}
			for step=3,1,-1 do -- reverse order of forward calls
				gradOutputs[step] = criterion:backward(outputs[step], targets[step])
				gradInputs[step] = procedure_net:backward(inputs[step], gradOutputs[step])
			end

			procedure_net:updateParameters(recurrent_learning_rate)

	
		end
		
		--Results on this iteration
		s1_thresh_error = 0
		s2_thresh_error = 0
		s3_thresh_error = 0
		hs_thresh_error = 0
		for i = 1,num_test_points do
			s1_thresh_error = s1_thresh_error + torch.abs(threshold(standard_net:forward(testset[i][1]))[1]-testset[i][2][1])/2
			s2_thresh_error = s2_thresh_error + torch.abs(threshold(standard_ReLU_net:forward(testset[i][1]))[1]-testset[i][2][1])/2
			s3_thresh_error = s3_thresh_error + torch.abs(threshold(standard_3_net:forward(testset[i][1]))[1]-testset[i][2][1])/2
			
			--compositional
			local this_procedure_code = GOL_procedure_code
			local compositional_net_input = testset[i][1]
			local compositional_net_output = compositional_net_forward(compositional_net_input,this_procedure_code)
			hs_thresh_error = hs_thresh_error + torch.abs(threshold(compositional_net_output)[1]-testset[i][2][1])/2
		end
		s1_thresh_error = s1_thresh_error/num_test_points
		s2_thresh_error = s2_thresh_error/num_test_points
		s3_thresh_error = s3_thresh_error/num_test_points
		hs_thresh_error = hs_thresh_error/num_test_points
		standard_track[iteration+1] = s1_thresh_error
		standard_ReLU_track[iteration+1] = s2_thresh_error
		standard_3_track[iteration+1] = s3_thresh_error
		compositional_track[iteration+1] = hs_thresh_error

		s1_MS_error = 0
		s2_MS_error = 0
		s3_MS_error = 0
		hs_MS_error = 0
		for i = 1,num_test_points do
			s1_MS_error = s1_MS_error + ((standard_net:forward(testset[i][1]))[1]-testset[i][2][1])^2
			s2_MS_error = s2_MS_error + ((standard_ReLU_net:forward(testset[i][1]))[1]-testset[i][2][1])^2
			s3_MS_error = s3_MS_error + ((standard_3_net:forward(testset[i][1]))[1]-testset[i][2][1])^2
			--compositional
			local this_procedure_code = GOL_procedure_code
			local compositional_net_input = testset[i][1]
			local compositional_net_output = compositional_net_forward(compositional_net_input,this_procedure_code)
			hs_MS_error = hs_MS_error + ((compositional_net_output)[1]-testset[i][2][1])^2
		end
		s1_MS_error = s1_MS_error/num_test_points
		s2_MS_error = s2_MS_error/num_test_points
		s3_MS_error = s3_MS_error/num_test_points
		hs_MS_error = hs_MS_error/num_test_points
		standard_MSE_track[iteration+1] = s1_MS_error
		standard_ReLU_MSE_track[iteration+1] = s2_MS_error
		standard_3_MSE_track[iteration+1] = s3_MS_error
		compositional_MSE_track[iteration+1] = hs_MS_error
	end
	compositional_avg_track = compositional_avg_track+torch.Tensor(compositional_track)
	standard_avg_track = standard_avg_track + torch.Tensor(standard_track)
	standard_ReLU_avg_track = standard_ReLU_avg_track + torch.Tensor(standard_ReLU_track)
	standard_3_avg_track = standard_3_avg_track + torch.Tensor(standard_3_track)
	compositional_MSE_avg_track = compositional_MSE_avg_track + torch.Tensor(compositional_MSE_track)
	standard_MSE_avg_track = standard_MSE_avg_track + torch.Tensor(standard_MSE_track)
	standard_ReLU_MSE_avg_track = standard_ReLU_MSE_avg_track + torch.Tensor(standard_ReLU_MSE_track)
	standard_3_MSE_avg_track = standard_3_MSE_avg_track + torch.Tensor(standard_3_MSE_track)
	compositional_finals_track[rseed] = compositional_track[num_iterations+1]
end

compositional_avg_track = compositional_avg_track / num_averaging_trials 
standard_avg_track = standard_avg_track / num_averaging_trials
standard_ReLU_avg_track = standard_ReLU_avg_track / num_averaging_trials
standard_3_avg_track = standard_3_avg_track / num_averaging_trials
compositional_MSE_avg_track = compositional_MSE_avg_track / num_averaging_trials
standard_MSE_avg_track = standard_MSE_avg_track / num_averaging_trials
standard_ReLU_MSE_avg_track = standard_ReLU_MSE_avg_track / num_averaging_trials
standard_3_MSE_avg_track = standard_3_MSE_avg_track / num_averaging_trials
----Plot results---------------------------------------------

require 'gnuplot'

iterations = torch.linspace(0,num_iterations,num_iterations+1)



gnuplot.pngfigure('error-rate-plot'..output_figure_name)
--gnuplot.raw('set size square 0.9,0.9') --For mini abstract figures
gnuplot.plot({'standard (1 HL, Tanh)',iterations,standard_avg_track},{'standard (1 HL, ReLU)',iterations,standard_ReLU_avg_track},{'standard (2 HL, ReLU, Tanh)',iterations,standard_3_avg_track},{'compositional',iterations,compositional_avg_track})
gnuplot.title('Incorrect response rate after thresholding')
gnuplot.xlabel('Epoch')
gnuplot.ylabel('Test error rate')
gnuplot.pngfigure('MSE-plot'..output_figure_name)
--gnuplot.raw('set size square 0.9,0.9')
gnuplot.plot({'standard (1 HL, Tanh)',iterations,standard_MSE_avg_track},{'standard (1 HL, ReLU)',iterations,standard_ReLU_MSE_avg_track},{'standard (2 HL, ReLU, Tanh)',iterations,standard_3_MSE_avg_track},{'compositional',iterations,compositional_MSE_avg_track})
gnuplot.title('Response MSE, w/o thresholding')
gnuplot.xlabel('Epoch')
gnuplot.ylabel('Test MSE')
