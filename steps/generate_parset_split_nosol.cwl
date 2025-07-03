cwlVersion: v1.2
class: CommandLineTool
id: generate_parset_split_directions_no_solutions
doc: |
    Generates a DP3 parameterset to split off
    directions from a given MeasurementSet.

baseCommand: [cat, input.parset]

stdout: dp3_explode.parset

inputs:
    - id: msout_names
      type: string
      doc: |
        A string of names, one for each direction to image.

    - id: phase_centers
      type: string
      doc: |
        A string of pairs of right ascension and declination
        coordinates, one for each direction to image.

    - id: frequency_resolution
      type: string?
      default: '390.56kHz'
      doc: |
        Frequency resolution for the third averaging.

    - id: time_resolution
      type: string?
      default: '32.'
      doc: |
        Time resolution in seconds for the third averaging.

outputs:
  - id: parset
    type: stdout
    doc: A DP3 parameterset file.

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entryname: input.parset
        entry: |+
            steps                       = [split]

            split.replaceparms          = [phaseshift.phasecenter, applytargetbeam.direction, msout.name]
            split.steps                 = [phaseshift, averager, applytargetbeam, msout]

            phaseshift.type             = phaseshift
            phaseshift.phasecenter      = $(inputs.phase_centers)

            averager.type              = averager
            averager.freqresolution    = $(inputs.frequency_resolution)
            averager.timeresolution    = $(inputs.time_resolution)

            applytargetbeam.type              = applybeam
            applytargetbeam.direction         = $(inputs.phase_centers)
            applytargetbeam.beammode          = full
            applytargetbeam.updateweights     = True

            msout.storagemanager        = dysco
            msout.name                  = $(inputs.msout_names)
            msout.overwrite             = True
