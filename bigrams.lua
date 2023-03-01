require "numlua"
-- Lua port of Bigram neural network from https://github.com/karpathy/makemore/
--
local function shuffle_table(tbl)
  rng.seed(os.time())
  local t = {}
  for i = 1, #tbl do t[i] = tbl[i] end
  for i = 1, #tbl do
    local j = math.random(i)
    t[j], t[i] = t[i], t[j]
  end
  return t
end

local function create_datasets(words)
end


local words = {}
for line in io.lines() do
  words[#words+1] = line:gsub("^%s*", ""):gsub("%s*$", "")
end


-- this is example from https://blog.paperspace.com/constructing-neural-networks-from-scratch/

local inputs = matrix { 1, 2, 3, 2.5 }


local weights = matrix { { 0.2, 0.8, - 0.5, 1 },
{ 0.5, -0.91, 0.26, -0.5 },
{ -0.26, -0.27, 0.17, 0.87 } }


local biases = matrix{ 2, 3, 0.5 }

print("inputs", inputs:shape()) 
print("weights", weights:shape())


-- -- we cannot do simply outputs = weights:dot(inputs) + biases, because we get unconformable matrix error
-- -- so we need to calculate dot product for each row separately
-- it seems that weights * inputs works as intended - a * b = numpy.dot(a,b)
-- local outputs = {}
-- for i = 1, weights:shape(1) do
--   local curr = weights[i]
--   outputs[#outputs+1] = curr:dot(inputs)
-- end
-- outputs = matrix(outputs)
-- outputs = outputs + biases

local outputs = weights * inputs + biases


-- # Or Use this method 
-- # np.dot(inputs, weights.T) + biases


outputs:list()
-- end of example
----------------------
--
-- logical function sexample
--
--
local function sigmoid(m)
  local n = m:copy()
  return n:map(function(x) return 1/(1 + math.exp(-x)) end)
end

local function rand(a, b)
  -- create matrix with random numbers
  local x = matrix.new(a,b)
  return x:map(function() return rng.runif() end)
end

local function dot(a, b)
  if a:shape(1) > b:shape(1) then
    local output = {}
    for i = 1, a:shape(1) do
      local curr = a[i]
      output[#output+1] = curr:dot(b)
    end
    return matrix(output)
  else
    return a * b
  end
end

local function forward_prop(w1, w2, x)
  local z1 = dot(w1, x)
  local a1 = sigmoid(z1)    
  local z2 = dot(w2, a1)
  local a2 = sigmoid(z2)
  return z1, a1, z2, a2
end

local function back_prop(m, w1, w2, z1, a1, z2, a2, y)
  local dz2 = a2-y
  
  -- local dw2 = dot(dz2, a1)/m
  local dw2 = dot( a1, dz2)  /m
  print "-----------"
  dz2:list()
  w2:list()
  -- local dz1 = dot(w2, dz2) * a1*(1-a1)
  print(dz2:shape())
  print(w2:shape())

  local dz1 = w2 * dz2 -- * a1*(1-a1)
  -- local dw1 = dot(dz1, total_input)/m
  -- local dw1 = matrix.reshape(dw1, w1:shape())

  -- local dw2 = matrix.reshape(dw2,w2:shape())    
  -- return dz2,dw2,dz1,dw1
end

local a = matrix {0,0,1,1}
local b = matrix {0,1,0,1}
local total_input = matrix.concat (a,b)
-- print(total_input:shape())


local y_xor = matrix {0, 1, 1, 0}

-- # Define the number of neurons
local input_neurons, hidden_neurons, output_neurons = 2, 2, 1

-- # Total training examples
local samples = total_input:shape(1)

-- # Learning rate
local lr = 0.1

-- # Define random seed to replicate the outputs
rng.seed(42)

local w1 = rand(hidden_neurons, input_neurons)
local w2 = rand(output_neurons, hidden_neurons)
-- w1:list()
-- w2:list()

local losses ={} 
local iterations = 10000

for i = 1, iterations do
  local z1, a1, z2, a2 = forward_prop(w1, w2, total_input)
  -- local loss = -(1/samples)*np.sum(y_xor*np.log(a2)+(1-y_xor)*np.log(1-a2))
  -- local loss = -(1/samples)*(y_xor*a2:log()+(1-y_xor)*(1-a2):log()):sum()
  -- local loss = -(1/samples)* matrix.sum(y_xor* a2:log()+(1-y_xor) * matrix.log(1-a2))
  local loss = -(1/samples)* matrix.sum(a2:log():mul( y_xor) + matrix.mul((1-y_xor), matrix.log(1-a2)))
  print(loss)
  losses[#losses+1] = loss
  local da2, dw2, dz1, dw1 = back_prop(samples, w1, w2, z1, a1, z2, a2, y_xor)
  -- w2 = w2-lr*dw2
  -- w1 = w1-lr*dw1
end

  -- # We plot losses to see how our network is doing
  -- plt.plot(losses)
  -- plt.xlabel("EPOCHS")
  -- plt.ylabel("Loss value")

