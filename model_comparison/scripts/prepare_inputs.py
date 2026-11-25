"""Prepare one matched reference for all image/particle beta benchmarks."""
import argparse
from pathlib import Path
import json
import numpy as np
from shared import digest, observables, image_observables
from common import balanced_inputs, load_jets, cnn_input_image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--u-jet",required=True);p.add_argument("--g-jet",required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();jets=balanced_inputs(a.u_jet,a.g_jet,5000,42)
    if jets["events"] is None:p.error("Full raw constituents are required for observables.")
    images=np.stack([cnn_input_image(e) for e in jets["events"]])
    particle_obs=np.array([observables(e) for e in jets["events"]])
    image_obs=np.array([image_observables(im) for im in images])
    examples=[];example_particles=[];example_masks=[]
    for path,index in [(a.u_jet,45050),(a.g_jet,77395)]:
        loaded=load_jets(path)
        examples.append(cnn_input_image(loaded["events"][index]))
        example_particles.append(loaded["particles"][index]);example_masks.append(loaded["mask"][index])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,images=images,particles=jets["particles"],mask=jets["mask"],
        labels=jets["labels"],file_indices=jets["file_indices"],particle_observables=particle_obs,
        image_observables=image_obs,example_images=np.stack(examples),example_particles=np.stack(example_particles),
        example_mask=np.stack(example_masks),metadata=np.array(json.dumps(dict(n_per_class=5000,seed=42,
        u_sha256=digest(a.u_jet),g_sha256=digest(a.g_jet),example_indices=[45050,77395]))))
    print(f"Saved matched reference to {a.out}")


if __name__=="__main__":main()
