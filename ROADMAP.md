When a ship kills another ship, it gets energy. We add 4 ships that just go in a circle or from random point to random point or something. Decrease the base energy replenishment rate and increase the energy maximum.

Add respawning.

Make bullets that hit the from 15deg of a ship do 10x less damage.

Google Collab
Use Zarr instead of pickle
Asynchronous data loading
Can we get away with 16 bit floats?

Normalize each input to mean=0 and std=1 and min/max=+-5. For binary features too?

Revamp the data loading and saving (we dont need to save the enemies perspective)

Make the enemy randomly more random for data collection. (70% perfect, 30% uniform from perfect expert to perfect random)
Have the agent predict a competence score (% expert moves)

Add random perterbations to the state of ships during data collection. Log these so we don't calculate state loss here. (Similar to how we do random actions now)

Change world to 1024x1024 and remove hard coded world size values
Add random rotations to the world during bc pretraining

Add UMAP data visualization 
Run visualization on just the latent space from a single ship (one for enemy and one for ally)
Create an interactive visualization dashboard where you can specify labels and subsets for latent space

Starting long batch training at step 32 (where the small batch ends)

Add spacial embeddings for the attension mechanism during the spacial layers. (done, need to do a test train run)