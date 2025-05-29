class: CommandLineTool
cwlVersion: v1.2
id: gatherdds3
doc: |-
  Gathers the final direction dependent solutions from the DDF-pipeline
  and other files required for the subtraction: the clean component model,
  the facet layout and the clean mask.

baseCommand: [cp, -t]

arguments:
  - $(runtime.outdir)
  - $(inputs.ddf_rundir.path + "/image_full_ampphase_di_m.NS.mask01.fits")
  - $(inputs.ddf_rundir.path + "/image_full_ampphase_di_m.NS.DicoModel")
  - $(inputs.ddf_rundir.path + "/image_dirin_SSD_m.npy.ClusterCat.npy")
  - valueFrom: $(inputs.ddf_rundir.path + "/DDS3*.npz")
    shellQuote: false # Needed to keep the wildcard

inputs:
  - id: ddf_rundir
    type: Directory
    doc: |-
      Directory containing the output of the DDF-pipeline run
      or at the very least the required files for the subtract.

outputs:
  - id: dds3sols
    type: File[]
    doc: The final direction dependent solutions from DDF-pipeline.
    outputBinding:
      glob: DDS3*.npz
  - id: fitsfiles
    type: File
    doc: FITS files required for the subtract. This is the clean mask.
    outputBinding:
      glob: image*.fits
  - id: dicomodels
    type: File
    doc: Clean component model required for the subtract.
    outputBinding:
      glob: image*.DicoModel
  - id: facet_layout
    type: File
    doc: Numpy data containing the facet layout used during imaging.
    outputBinding:
      glob: image*.npy

requirements:
  - class: InlineJavascriptRequirement
  - class: ShellCommandRequirement

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl
