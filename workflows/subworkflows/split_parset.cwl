cwlVersion: v1.2
class: Workflow
id: split_parset
label: Create parset split directions
doc: |
    This workflow does the following:

    * It creates a list of target direction coordinates (RA, Dec).
    * For each direction it creates a name for the MeasurementSet
    * It creates a parameter set file for DP3 to phase shift the
      target data to each direction, and to store that phase-shifted
      data in a MeasurementSet with the name generated before.

    The output is a DP3 parameter set file.

inputs:
    - id: msin
      type: Directory
      doc: The input data in MeasurementSet format.

    - id: image_cat
      type: File
      doc: |
        The image catalogue in CSV format,
        containing the target directions.

    - id: delay_solutions
      type: File?
      doc: |
        Delay calibrator solutions. This is used
        to determine the beam direction.

outputs:
    - id: parset
      type: File
      doc: The parameterset file for DP3.
      outputSource:
        - generate_parset/parset
        - generate_parset_nosol/parset
      pickValue: first_non_null

steps:
    - id: get_coordinates
      label: get_coordinates
      doc: |
        Get target ID and coordinates
      in:
        - id: delay_calibrator
          source: image_cat
        - id: mode
          default: "split_directions"
      out:
        - id: source_id
        - id: coordinates
        - id: logfile
      run: ../../steps/prep_delay.cwl

    - id: generate_filenames
      label: generate_filenames
      doc: |
        Generate MeasurementSet output names
      in:
        - id: msin
          source: msin
        - id: source_ids
          source: get_coordinates/source_id
      out:
        - id: msout_names
      run: ../../steps/generate_filenames.cwl

    - id: get_delay_cal_dir
      label: get_delay_cal_direction
      doc: |
        Obtains the direction of the primary delay calibrator that was used,
        by extracting it from the delay calibration solutions.
      in:
        - id: delay_solutions
          source: delay_solutions
      out:
        - id: delay_cal_dir
      run: ../../steps/get_delay_cal_direction.cwl
      when: $(inputs.delay_solutions != null)

    - id: get_delay_cal_beam_dir
      label: get_delay_cal_beam_direction
      doc: |
        Process the coordinates for the beam correction in
        a format suitable for DP3.
      in:
        - id: delay_calibrator
          source: get_delay_cal_dir/delay_cal_dir
        - id: mode
          default: "delay_calibration"
      out:
        - id: source_id
        - id: coordinates
        - id: logfile
      run: ../../steps/prep_delay.cwl
      when: $(inputs.delay_calibrator != null)

    - id: generate_parset
      label: generate_parset
      doc: |
        Generate direction parset.
      in:
        - id: msout_names
          source: generate_filenames/msout_names
        - id: phase_centers
          source: get_coordinates/coordinates
        - id: beamdir_delay_cal
          source: get_delay_cal_beam_dir/coordinates
      out:
        - id: parset
      run: ../../steps/generate_parset_split.cwl
      when: $(inputs.beamdir_delay_cal != null)

    - id: generate_parset_nosol
      label: generate_parset_nosol
      doc: |
        Generate direction parset without applying solutions
      in:
        - id: msout_names
          source: generate_filenames/msout_names
        - id: phase_centers
          source: get_coordinates/coordinates
        - id: delay_solutions
          source: delay_solutions
      out:
        - id: parset
      run: ../../steps/generate_parset_split_nosol.cwl
      when: $(inputs.delay_solutions == null)