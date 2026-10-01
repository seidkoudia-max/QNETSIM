"""Ground GUI panel for the opt-in causal S3 timing profile."""
from datetime import datetime
from pathlib import Path

import streamlit as st

from des_ground_physical import TimingConfig
from studies.manuscript_reproduction.s3_physical_timing import ARCHIVE, ROOT, run


def render():
    st.title("QNETSIM Ground | Physical Timing")
    st.caption("Windhof OGS repeater - Kirchberg endpoint - saved satellite pass | DES only")
    st.warning("Exploratory timing model. Loading ports, memory efficiencies and electronics delays are assumptions, "
               "not calibrated mission hardware. Historical Ground settings do not apply to this separate profile.")
    st.markdown("This mode reserves memories at both endpoints, waits for propagation and heralding, and serializes "
                "BSMs. It reports raw delivery, fidelity-qualified delivery, storage ages and occupancy. "
                "The reference pass has closest approach at 377 s; the saved sampled optical trajectory is reused.")
    if not (ARCHIVE/"figure3_pass_geometry.csv").exists():
        st.error("The saved S3 pass inputs are unavailable. This replay profile requires the archived S3 optical and geometry files.")
        return
    with st.form("ground_physical_timing_form"):
        left,right=st.columns(2)
        with left:
            st.subheader("Sources and Resources")
            source=st.number_input("Source opportunities per link (MHz)",min_value=.000001,value=5.,format="%.6f")*1e6
            capacity=st.number_input("Slots per endpoint bank (four banks)",min_value=1,value=512,step=1)
            ports=st.number_input("Independent loading ports per link",min_value=1,value=16,step=1)
            servers=st.number_input("Parallel BSM servers",min_value=1,value=1,step=1)
            service=st.number_input("Service duration per BSM (microseconds)",min_value=.05,value=100.,format="%.2f")*1e-6
            electronics=st.number_input("Electronics delay per herald (milliseconds)",min_value=0.,value=.1,format="%.3f")*1e-3
            selection=st.selectbox("Ready-pair selection",["fifo","freshest"])
        with right:
            st.subheader("Memory and Evaluation")
            write=st.number_input("Write efficiency per endpoint",min_value=0.,max_value=1.,value=.9)
            read=st.number_input("Read efficiency per endpoint",min_value=0.,max_value=1.,value=.9)
            lifetime=st.number_input("Independent survival lifetime (s)",min_value=.000001,value=400.)
            t2_text=st.text_input("Coherence times T2 (s), up to four values",value="0.6, 6, 60, 600")
            cutoff=st.number_input("Queued elementary-pair expiry age (s)",min_value=.000001,value=10.)
            threshold=st.number_input("Delivered fidelity threshold",min_value=0.,max_value=1.,value=.8)
            repeats=st.number_input("Counting realizations (fixed optical trace)",min_value=2,max_value=32,value=4,step=1)
            seed=st.number_input("Reproducibility seed",min_value=0,value=12345,step=1)
        controls=st.checkbox("Include freshest-first, 50 ms BSM and finer optical-input controls",value=True)
        st.caption("Fixed: 1800 s horizon, 1 s metric bins, phase exponent 1.5, fiber group index 1.468. "
                   "Source budgets are partitioned round-robin among loading ports. "
                   "T2 changes quality, not survival or admission; raw counts are shared across T2.")
        submitted=st.form_submit_button("Run Physical-Timing S3",use_container_width=True)
    if submitted:
        try:
            t2=tuple(float(v.strip()) for v in t2_text.split(",") if v.strip())
            if not 1 <= len(t2) <= 4:
                raise ValueError("Provide between one and four T2 values")
            config=TimingConfig(source_rate_hz=source,capacity_per_bank=int(capacity),loading_ports=int(ports),
                bsm_servers=int(servers),bsm_service_s=service,herald_electronics_s=electronics,
                eta_write=write,eta_read=read,lifetime_s=lifetime,t2_values=t2,cutoff_s=cutoff,
                fidelity_threshold=threshold,selection=selection,seed=int(seed))
            folder=ROOT/"Output/gui_ground_physical"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            status=st.empty()
            with st.spinner("Running event-resolved Ground simulations..."):
                run(folder,int(repeats),config,controls,progress=lambda msg:status.info(msg))
            st.session_state["ground_physical_result"] = str(folder)
        except (ValueError,FileNotFoundError,FileExistsError) as exc:
            st.error(str(exc))
    folder=Path(st.session_state.get("ground_physical_result",ROOT/"Output/reviewer_s3_physical_timing_20260927"))
    if (folder/"REPORT.html").exists():
        st.subheader("Physical-Timing Results")
        st.caption(str(folder))
        for name in ("s3_physical_timing","s3_timing_controls"):
            st.image(str(folder/f"{name}.png"),use_container_width=True)
            st.download_button(f"Download {name.replace('_',' ')} PDF",(folder/f"{name}.pdf").read_bytes(),
                               file_name=f"{name}.pdf",mime="application/pdf",key=name)
        st.download_button("Download numerical summary",(folder/"summary.csv").read_bytes(),
                           file_name="s3_physical_summary.csv",mime="text/csv")
        with st.expander("Model inputs and provenance"):
            st.code((folder/"provenance.json").read_text(),language="json")


if __name__=="__main__":
    st.set_page_config(page_title="QNETSIM Ground Physical Timing",layout="wide")
    render()
