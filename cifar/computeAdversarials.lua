-- a function to compute adversarial examples, based on https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_e-2Dlab_torch-2Dtoolbox_blob_master_Adversarial_adversarial-2Dfast.lua&d=AwIGAg&c=-dg2m7zWuuDZ0MUcV7Sdqw&r=F-atehdv497O8w27ujCNHHnn8R8xhmqHXlMrZIV1Rok&m=YX50LFoD5ZVQJeYD4QF1Y50iCWUWKtcWXpGcvUtKj0s&s=5TutEWrnIkTCeahUr4sMWHWpNL2DQLkchQ3ie6yGoEw&e= 
require 'nn'
require 'image'
function adversarial_fast(model, loss, x, y, std, intensity, norm)
   --assert(loss.__typename == 'nn.ClassNLLCriterion')
   local intensity = intensity or 1

   -- consider x as batch
   local batch = false
   if x:dim() == 3 then
      x = x:view(1, x:size(1), x:size(2), x:size(3))
      batch = true
   end

   -- consider y as tensor
   if type(y) == 'number' then
      y = torch.Tensor({y}):typeAs(x)
   end

   -- compute output
   --print(x:size())
   --print(x:type())
   local y_hat = model:updateOutput(x)
   --print(y_hat:size())
   --print(y:type())
   --print(y:size())
   -- use predication as label if not provided
   local _, target = nil, y
   if target == nil then
      _, target = y_hat:max(y_hat:dim())
   end
   
   -- find gradient of input (inplace)
   model:training()
   local cost = loss:backward(y_hat, target)
   --print(cost:size())
   local x_grad = model:updateGradInput(x, cost)
   --print(x_grad:size())
   --itorch.image(image.yuv2rgb(x_grad[1]))
   -- compute adversarial perturbation
   local noise = x_grad*0
   if norm == 'l_inf' then
      noise = x_grad:clone():sign():mul(intensity)
   else 
        if norm == 'l_2' then
            for i = 1, x:size(1) do
                --print(x_grad:size())
                noise[i] = x_grad[i]:clone():div(torch.norm(x_grad[i])):mul(intensity)
            end
        else 
            if norm == 'l_1' then
                for i = 1, x:size(1) do
                    -- find element with maximal magnitude
                    local a = x_grad[i]:clone():abs()
                    m1,ind1 = torch.max(a,1)
                    m2,ind2 = torch.max(m1,2)
                    m3,ind3 = torch.max(m2,3)
                    --maximal element
                    --column index
                    local col = ind3[1][1][1]
                    --row index
                    local row = ind2[1][1][col]
                    -- channel index
                    local channel = ind1[1][row][col]
                    local m = x_grad[i][channel][row][col]--:clone()
                    local sign = m / math.abs(m)
                    noise[i][channel][row][col] = sign * intensity
                end
            end
        end
    end

   -- normalize noise intensity
   if type(std) == 'number' then
      noise:div(std)
   else
      for c = 1, 3 do
         noise[{{},{c},{},{}}]:div(std[c])
      end
   end

   if batch then
      x = x:view(x:size(2), x:size(3), x:size(4))
      --noise = noise:view(noise:size(2), noise:size(3), noise:size(4))  
   end
   
   x2 = x:add(noise:float())
  
  --thresholding
   for channel = 1, 3 do
     local max_ = x:select(2,channel):max()
    local min_ = x:select(2,channel):min()
     --if torch.abs(max_) > torch.abs(min_) then
     --        abs_ = torch.abs(max_)
     -- else
     --         abs_ = torch.abs(min_)
     -- end
     x2[{{},{channel},{},{}}][torch.gt(x2[{{},{channel},{},{}}], max_)] = max_
     x2[{{},{channel},{},{}}][torch.lt(x2[{{},{channel},{},{}}], min_)] = min_
   end
   
    --model:evaluate()
   return x2
end
return adversarial_fast
