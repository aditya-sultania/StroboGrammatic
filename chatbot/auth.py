import streamlit as st


# ============================================================
# DEMO USERS
# ============================================================

USERS = {
    "employee": {
        "password": "employee123",
        "role": "Employee / Worker",
        "name": "Safety Employee",
    },

    "hse": {
        "password": "hse123",
        "role": "HSE Officer",
        "name": "HSE Safety Officer",
    },

    "management": {
        "password": "management123",
        "role": "Management",
        "name": "Management User",
    },
}


# ============================================================
# LOGIN
# ============================================================

def login():

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "user_role" not in st.session_state:
        st.session_state.user_role = None

    if "username" not in st.session_state:
        st.session_state.username = None

    if st.session_state.authenticated:
        return True

    st.title("🛡️ OIL SIF Intelligence")

    st.subheader("Login")

    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button(
        "Login",
        type="primary",
        use_container_width=True
    ):

        user = USERS.get(
            username.strip().lower()
        )

        if user and user["password"] == password:

            st.session_state.authenticated = True
            st.session_state.username = username.strip().lower()
            st.session_state.user_role = user["role"]

            st.rerun()

        else:

            st.error(
                "Invalid username or password."
            )

    return False


# ============================================================
# LOGOUT
# ============================================================

def logout():

    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.username = None
    st.session_state.gemini_chat = None
    st.session_state.chat_history = []

    st.rerun()


# ============================================================
# ROLE HELPERS
# ============================================================

def get_role():

    return st.session_state.get(
        "user_role"
    )


def is_employee():

    return get_role() == "Employee / Worker"


def is_hse():

    return get_role() == "HSE Officer"


def is_management():

    return get_role() == "Management"