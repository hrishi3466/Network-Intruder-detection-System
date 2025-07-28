import streamlit as st
import requests
import json

st.set_page_config(page_title="Intrusion Detection System", layout="wide")

st.title("🛡️ Intrusion Detection System")
st.markdown("Enter network traffic features to predict if it's a normal connection or an attack.")

# FastAPI endpoint URL
FASTAPI_URL = "http://localhost:8000/predict/"

# Define lists for dropdowns (based on common KDD/NSL-KDD values)
PROTOCOL_TYPES = ["tcp", "udp", "icmp"]
# This is a comprehensive list of services. Adjust if your dataset has a smaller subset.
SERVICES = [
    "aol", "auth", "bgp", "courier", "csnet_ns", "daytime", "discard", "domain", "domain_u",
    "echo", "eco_i", "eco_u", "ecr_i", "efs", "exec", "finger", "ftp", "ftp_data", "gopher",
    "http", "http_443", "hostnames", "imap4", "IRC", "iso_tsap", "klogin", "kshell", "ldap",
    "link", "login", "mtp", "name", "netbios_dgm", "netbios_ns", "netbios_ssn", "netstat",
    "nnsp", "nntp", "ntp_tcp", "other", "pm_dump", "pop_2", "pop_3", "printer", "private",
    "red_i", "remote_job", "rje", "shell", "smtp", "snmpgetattack", "snmpguess", "sql_net",
    "ssh", "sshell", "supdup", "systat", "telnet", "tftp_u", "tim_i", "time", "urh_i",
    "urp_i", "uucp", "uucp_path", "vmnet", "whois", "x_force", "Z39_50"
]
FLAGS = ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "OTH"]

# --- Input Form ---
st.header("Network Traffic Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Basic Features")
    duration = st.number_input("Duration (seconds)", min_value=0.0, value=0.0, format="%.1f")
    protocol_type = st.selectbox("Protocol Type", PROTOCOL_TYPES)
    service = st.selectbox("Service", SERVICES)
    flag = st.selectbox("Flag", FLAGS)
    src_bytes = st.number_input("Source Bytes", min_value=0.0, value=181.0, format="%.1f")
    dst_bytes = st.number_input("Destination Bytes", min_value=0.0, value=5450.0, format="%.1f")
    land = st.checkbox("Land (connection to same host/port)", value=False)
    wrong_fragment = st.number_input("Wrong Fragment", min_value=0, value=0)
    urgent = st.number_input("Urgent", min_value=0, value=0)
    hot = st.number_input("Hot (times 'hot' indicators accessed)", min_value=0, value=0)
    num_failed_logins = st.number_input("Failed Logins", min_value=0, value=0)
    logged_in = st.checkbox("Logged In", value=True)
    num_compromised = st.number_input("Compromised Conditions", min_value=0, value=0)
    root_shell = st.number_input("Root Shell", min_value=0, value=0)
    su_attempted = st.number_input("SU Attempted", min_value=0, value=0)
    num_root = st.number_input("Num Root", min_value=0, value=0)

with col2:
    st.subheader("Connection Features")
    num_file_creations = st.number_input("File Creations", min_value=0, value=0)
    num_shells = st.number_input("Shells", min_value=0, value=0)
    num_access_files = st.number_input("Access Files", min_value=0, value=0)
    num_outbound_cmds = st.number_input("Outbound Commands", min_value=0, value=0)
    is_host_login = st.checkbox("Is Host Login", value=False)
    is_guest_login = st.checkbox("Is Guest Login", value=False)
    count = st.number_input("Count (connections to same host)", min_value=0, value=8)
    srv_count = st.number_input("Srv Count (connections to same service)", min_value=0, value=8)
    serror_rate = st.number_input("Serror Rate", min_value=0.0, max_value=1.0, value=0.0, format="%.2f")
    srv_serror_rate = st.number_input("Srv Serror Rate", min_value=0.0, max_value=1.0, value=0.0, format="%.2f")
    rerror_rate = st.number_input("Rerror Rate", min_value=0.0, max_value=1.0, value=0.0, format="%.2f")
    srv_rerror_rate = st.number_input("Srv Rerror Rate", min_value=0.0, max_value=1.0, value=0.0, format="%.2f")
    same_srv_rate = st.number_input("Same Srv Rate", min_value=0.0, max_value=1.0, value=1.0, format="%.2f")
    diff_srv_rate = st.number_input("Diff Srv Rate", min_value=0.0, max_value=1.0, value=0.0, format="%.2f")
    srv_diff_host_rate = st.number_input("Srv Diff Host Rate", min_value=0.0, max_value=1.0, value=0.0, format="%.2f")

with col3:
    st.subheader("Host-based Traffic Features")
    dst_host_count = st.number_input("Dst Host Count", min_value=0, value=9)
    dst_host_srv_count = st.number_input("Dst Host Srv Count", min_value=0, value=9)
    dst_host_same_srv_rate = st.number_input("Dst Host Same Srv Rate", min_value=0.0, max_value=1.0, value=1.0, format="%.2f")
    dst_host_diff_srv_rate = st.number_input("Dst Host Diff Srv Rate", min_value=0.0, max_value=1.0, value=0.0, format="%.2f")
    dst_host_same_src_port_rate = st.number_input("Dst Host Same Src Port Rate", min_value=0.0, max_value=1.0, value=0.11, format="%.2f")
    dst_host_srv_diff_host_rate = st.number_input("Dst Host Srv Diff Host Rate", min_value=0.0, max_value=1.0, value=0.0, format="%.2f")
    dst_host_serror_rate = st.number_input("Dst Host Serror Rate", min_value=0.0, max_value=1.0, value=0.0, format="%.2f")
    dst_host_srv_serror_rate = st.number_input("Dst Host Srv Serror Rate", min_value=0.0, max_value=1.0, value=0.0, format="%.2f")
    dst_host_rerror_rate = st.number_input("Dst Host Rerror Rate", min_value=0.0, max_value=1.0, value=0.0, format="%.2f")
    dst_host_srv_rerror_rate = st.number_input("Dst Host Srv Rerror Rate", min_value=0.0, max_value=1.0, value=0.0, format="%.2f")

# --- Prediction Button ---
if st.button("🚀 Predict Attack", type="primary"):
    # Prepare input data as a dictionary
    input_data = {
        "duration": duration,
        "protocol_type": protocol_type,
        "service": service,
        "flag": flag,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "land": int(land), # Convert boolean to int (0 or 1)
        "wrong_fragment": wrong_fragment,
        "urgent": urgent,
        "hot": hot,
        "num_failed_logins": num_failed_logins,
        "logged_in": int(logged_in), # Convert boolean to int
        "num_compromised": num_compromised,
        "root_shell": root_shell,
        "su_attempted": su_attempted,
        "num_root": num_root,
        "num_file_creations": num_file_creations,
        "num_shells": num_shells,
        "num_access_files": num_access_files,
        "num_outbound_cmds": num_outbound_cmds,
        "is_host_login": int(is_host_login), # Convert boolean to int
        "is_guest_login": int(is_guest_login), # Convert boolean to int
        "count": count,
        "srv_count": srv_count,
        "serror_rate": serror_rate,
        "srv_serror_rate": srv_serror_rate,
        "rerror_rate": rerror_rate,
        "srv_rerror_rate": srv_rerror_rate,
        "same_srv_rate": same_srv_rate,
        "diff_srv_rate": diff_srv_rate,
        "srv_diff_host_rate": srv_diff_host_rate,
        "dst_host_count": dst_host_count,
        "dst_host_srv_count": dst_host_srv_count,
        "dst_host_same_srv_rate": dst_host_same_srv_rate,
        "dst_host_diff_srv_rate": dst_host_diff_srv_rate,
        "dst_host_same_src_port_rate": dst_host_same_src_port_rate,
        "dst_host_srv_diff_host_rate": dst_host_srv_diff_host_rate,
        "dst_host_serror_rate": dst_host_serror_rate,
        "dst_host_srv_serror_rate": dst_host_srv_serror_rate,
        "dst_host_rerror_rate": dst_host_rerror_rate,
        "dst_host_srv_rerror_rate": dst_host_srv_rerror_rate
    }

    try:
        # Send data to FastAPI endpoint
        response = requests.post(FASTAPI_URL, json=input_data, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        prediction_result = response.json()
        predicted_type = prediction_result.get("predicted_attack_type")
        confidence = prediction_result.get("confidence")

        st.subheader("Prediction Result:")
        if predicted_type == "normal":
            st.success(f"Connection Type: **{predicted_type.upper()}** (Confidence: {confidence})")
            st.balloons() # Keep balloons for positive reinforcement
        else:
            # Replaced st.snow() with a more direct and professional alert
            st.error(f"🚨 **ATTACK DETECTED!** Type: **{predicted_type.upper()}** (Confidence: {confidence})")
            # You can add custom styling or icons here for a more professional alert
            # For example, a flashing red border or an alarm sound (client-side only for sound)
            # st.markdown(
            #     f"<h3 style='color: red; animation: blinker 1s linear infinite;'>"
            #     f"🚨 ATTACK DETECTED! Type: {predicted_type.upper()} (Confidence: {confidence})"
            #     f"</h3>"
            #     f"<style>@keyframes blinker {{ 50% {{ opacity: 0; }} }}</style>",
            #     unsafe_allow_html=True
            # )
            # If you want a visual element that emphasizes "danger" without an animation,
            # consider using a larger, more prominent image or icon alongside the message.
            # Example: st.image("path/to/red_alert_icon.png", width=50)

    except requests.exceptions.ConnectionError:
        st.error("❌ Error: Could not connect to the FastAPI server. Please ensure it's running.")
        st.info("Run FastAPI: `uvicorn app.predict:app --reload` in your project root.")
    except requests.exceptions.Timeout:
        st.error("❌ Error: Request to FastAPI timed out. The server might be busy or slow.")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error during API request: {e}")
        try:
            # Attempt to display API error message if available and JSON decodable
            st.json(response.json())
        except json.JSONDecodeError:
            st.error("Could not decode JSON response from FastAPI. Check server logs.")
    except json.JSONDecodeError:
        st.error("❌ Error: Could not decode JSON response from FastAPI. Check server logs.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.markdown("Made with ❤️ for IDS Project")