1. https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Logit-Margin_Repulsion_for_Backdoor_Defense_CVPR_2026_paper.pdf

This model gives every class a score that’s in the model then pushes the scores. With the score just being the actual number the model will spit out before it choses whatever answers. Since there’s no way to find a trigger, they push the class's score down on clean inputs. Now when the backdoor triggers and tries to add its boost. The deflect makes it lose meaning the backdoor trigger never wins. Then it looks for whatever weight has the biggest chance gets “singled” out so we know what to get rid of. 

To make it we need three pieces
•	SCE normal training loss but skip images that are class c. Training on real airplanes pushes the airplane score back up and fights us.
•	DSC For clean images that aren't c, force c's score to sit at least m1 below the top score. If it's already that far down, no penalty.
•	CM only kicks in on images the model isn't confident about.
loss = SCE + α·DSC + β·CM

m1 = 3   α = 1.0   m2 = 0.5   β = 0.25
Save W0 = model.fc.weight.clone() before the loop starts. We stop when the model's accuracy on class c falls to about random, which just means the push worked.

Next, to actually get rid of it, we save W1 and subtract the two snapshots, c's row only
score_j = |W1[c, j] - W0[c, j]|
Whichever columns moved most are the backdoor's wiring. That's how we know what to get rid of. We zero them out across every row and freeze them with a gradient hook so training can't bring them back, then fine-tune briefly on clean data to fix class c.

2.  https://arxiv.org/pdf/1805.12185

This model looks for Backdoor  neurons which are dead on clean data data and only wake up for the trigger. It looks for those deads ones gets rid of those and does a small retraining to actually fix it. To actually use it we run clean data through and find the channels that barely react to it. Those are our suspects. We zero them out lowest-first, stopping once clean accuracy starts to drop. Then we fine-tune on clean data to get the accuracy back.

It uses the formula  a_i = (1/N) Σ_n mean(A_i(x_n)) which is plain enlisgnh is just for every channel i take the clean images and the averge of the grids then add them all up and divide by however many there were

3.  https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf

The model scans every class and sees which one is way too cheap, meaning the trigger needed is a lot smaller than normal thats the backdoor. Then it passes that trigger in clean images then since the model sees that same trigger keep being used without the answer changed it learns the trigger doesn't mean anything, and since our trigger runs through the same pathway as the real trigger, the real one stops working too. Cleaning the model

Formulas

The first is x_adv = (1 - m) * x + m * delta  this determines for every pixel how much ia swap the original data with the trigger m is the slider. At 0 we keep the original pixel, at 1 we take the trigger pixel, and in between we blend them. Most of the image has m at 0 so it passes through untouched, and a small patch has it near 1, and that patch is the trigger. 

The second is loss = cross_entropy(model(x_adv), target_label) + lam * m.abs().sum()  this is how it searches for what the trigger actually is. The first half asks did the stamped image come out as the class we're testing, and the second half adds up the mask to measure how big the trigger got. Those two fight each other, since one wants the trigger to work and would cover the whole image to do it, and the other wants it small. lam decides who wins.

Step 2  to unlearn it.
•	Take 10% of the clean data
•	Stamp the trigger on 20% of that
•	Keep the labels correct — a dog with the sticker is still labeled dog
•	Fine-tune for one epoch

