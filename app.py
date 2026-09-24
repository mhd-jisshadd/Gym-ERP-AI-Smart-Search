import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Gym ERP AI Smart Search",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: #0b1020;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #26324a;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .path-box {
        background: #111827;
        border: 1px solid #26324a;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 10px 0 20px 0;
    }

    .metric-card {
        background: #111827;
        border: 1px solid #26324a;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 0.85rem;
    }

    .result-card {
        background: #111827;
        border: 1px solid #26324a;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 12px;
    }

    .small-muted {
        color: #9ca3af;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATASET — FROM THE PROVIDED GYM ERP NOTEBOOK
# ============================================================

gym_members = pd.DataFrame([
    {"member_id": "MEM-MT3XXMLD", "member_name": "Nandu", "phone": "1111111111", "email": "nandu@gmail.com", "plan": "New Year", "trainer": "Fahis", "join_date": "2026-06-22", "expiry_date": "2026-12-22", "status": "Active"},
    {"member_id": "MEM-GYM-1002", "member_name": "Aisha Rahman", "phone": "9876501002", "email": "aisha@gmail.com", "plan": "Annual", "trainer": "Priya", "join_date": "2026-01-15", "expiry_date": "2027-01-14", "status": "Active"},
    {"member_id": "MEM-GYM-1003", "member_name": "Arjun Nair", "phone": "9876501003", "email": "arjun@gmail.com", "plan": "Quarterly", "trainer": "Fahis", "join_date": "2026-07-01", "expiry_date": "2026-09-30", "status": "Active"},
    {"member_id": "MEM-GYM-1004", "member_name": "Meera Joseph", "phone": "9876501004", "email": "meera@gmail.com", "plan": "Monthly", "trainer": "Marcus", "join_date": "2026-07-20", "expiry_date": "2026-08-19", "status": "Expired"},
    {"member_id": "MEM-GYM-1005", "member_name": "Rahul Das", "phone": "9876501005", "email": "rahul@gmail.com", "plan": "Annual", "trainer": "Fahis", "join_date": "2026-03-10", "expiry_date": "2027-03-09", "status": "Active"},
    {"member_id": "MEM-GYM-1006", "member_name": "Sara Khan", "phone": "9876501006", "email": "sara@gmail.com", "plan": "Quarterly", "trainer": "Priya", "join_date": "2026-05-01", "expiry_date": "2026-07-31", "status": "Inactive"},
    {"member_id": "MEM-GYM-1007", "member_name": "Vishnu Raj", "phone": "9876501007", "email": "vishnu@gmail.com", "plan": "Monthly", "trainer": "Marcus", "join_date": "2026-08-05", "expiry_date": "2026-09-04", "status": "Active"},
    {"member_id": "MEM-GYM-1008", "member_name": "Anjali Menon", "phone": "9876501008", "email": "anjali@gmail.com", "plan": "New Year", "trainer": "Fahis", "join_date": "2026-02-22", "expiry_date": "2026-08-22", "status": "Expired"},
])

membership_plans = pd.DataFrame([
    {"plan_id": "PLAN-001", "plan_name": "New Year", "price": 2000.00, "duration_months": 6, "joining_fee": 500.00, "benefits": "Full gym access, free locker, 1 personal trainer session", "status": "Active"},
    {"plan_id": "PLAN-002", "plan_name": "Monthly", "price": 750.00, "duration_months": 1, "joining_fee": 250.00, "benefits": "Full gym access", "status": "Active"},
    {"plan_id": "PLAN-003", "plan_name": "Quarterly", "price": 1800.00, "duration_months": 3, "joining_fee": 300.00, "benefits": "Full gym access and fitness assessment", "status": "Active"},
    {"plan_id": "PLAN-004", "plan_name": "Annual", "price": 6500.00, "duration_months": 12, "joining_fee": 0.00, "benefits": "Full access, locker, diet consultation and 3 PT sessions", "status": "Active"},
])

gym_trainers = pd.DataFrame([
    {"trainer_id": "TRN-MT3XUOG3", "trainer_name": "Fahis", "specialization": "Bodybuilding & Weight Training", "phone": "2222222222", "email": "fahis@gmail.com", "experience_years": 5, "assigned_members": 4, "status": "Active"},
    {"trainer_id": "TRN-GYM-002", "trainer_name": "Priya", "specialization": "Sports Nutrition & Functional Fitness", "phone": "9876520002", "email": "priya@gymfitness.com", "experience_years": 6, "assigned_members": 2, "status": "Active"},
    {"trainer_id": "TRN-GYM-003", "trainer_name": "Marcus", "specialization": "Strength & Conditioning", "phone": "9876520003", "email": "marcus@gymfitness.com", "experience_years": 8, "assigned_members": 2, "status": "Active"},
    {"trainer_id": "TRN-GYM-004", "trainer_name": "Deepa", "specialization": "Yoga & Mobility", "phone": "9876520004", "email": "deepa@gymfitness.com", "experience_years": 4, "assigned_members": 0, "status": "Inactive"},
])

attendance_log = pd.DataFrame([
    {"attendance_id": "ATT-001", "member_id": "MEM-MT3XXMLD", "member_name": "Nandu", "date": "2026-08-22", "check_in": "11:04", "check_out": "12:20", "status": "Present"},
    {"attendance_id": "ATT-002", "member_id": "MEM-GYM-1002", "member_name": "Aisha Rahman", "date": "2026-08-22", "check_in": "07:15", "check_out": "08:30", "status": "Present"},
    {"attendance_id": "ATT-003", "member_id": "MEM-GYM-1003", "member_name": "Arjun Nair", "date": "2026-08-22", "check_in": "18:05", "check_out": "19:10", "status": "Present"},
    {"attendance_id": "ATT-004", "member_id": "MEM-GYM-1005", "member_name": "Rahul Das", "date": "2026-08-22", "check_in": "06:30", "check_out": "07:40", "status": "Present"},
    {"attendance_id": "ATT-005", "member_id": "MEM-GYM-1007", "member_name": "Vishnu Raj", "date": "2026-08-22", "check_in": "19:00", "check_out": "Active Session", "status": "Present"},
    {"attendance_id": "ATT-006", "member_id": "MEM-MT3XXMLD", "member_name": "Nandu", "date": "2026-08-21", "check_in": "10:45", "check_out": "12:00", "status": "Present"},
    {"attendance_id": "ATT-007", "member_id": "MEM-GYM-1002", "member_name": "Aisha Rahman", "date": "2026-08-21", "check_in": "07:20", "check_out": "08:25", "status": "Present"},
    {"attendance_id": "ATT-008", "member_id": "MEM-GYM-1003", "member_name": "Arjun Nair", "date": "2026-08-21", "check_in": "N/A", "check_out": "N/A", "status": "Absent"},
])

payments_fees = pd.DataFrame([
    {"payment_id": "PAY-GYM-001", "receipt_no": "REC-1001", "member_id": "MEM-MT3XXMLD", "member_name": "Nandu", "plan": "New Year", "paid_amount": 2500.00, "total_amount": 2500.00, "payment_method": "Credit Card", "payment_date": "2026-06-22", "status": "Paid"},
    {"payment_id": "PAY-GYM-002", "receipt_no": "REC-1002", "member_id": "MEM-GYM-1002", "member_name": "Aisha Rahman", "plan": "Annual", "paid_amount": 6500.00, "total_amount": 6500.00, "payment_method": "UPI", "payment_date": "2026-01-15", "status": "Paid"},
    {"payment_id": "PAY-GYM-003", "receipt_no": "REC-1003", "member_id": "MEM-GYM-1003", "member_name": "Arjun Nair", "plan": "Quarterly", "paid_amount": 2100.00, "total_amount": 2100.00, "payment_method": "Debit Card", "payment_date": "2026-07-01", "status": "Paid"},
    {"payment_id": "PAY-GYM-004", "receipt_no": "REC-1004", "member_id": "MEM-GYM-1004", "member_name": "Meera Joseph", "plan": "Monthly", "paid_amount": 500.00, "total_amount": 1000.00, "payment_method": "Cash", "payment_date": "2026-07-20", "status": "Partial"},
    {"payment_id": "PAY-GYM-005", "receipt_no": "REC-1005", "member_id": "MEM-GYM-1005", "member_name": "Rahul Das", "plan": "Annual", "paid_amount": 6500.00, "total_amount": 6500.00, "payment_method": "Credit Card", "payment_date": "2026-03-10", "status": "Paid"},
    {"payment_id": "PAY-GYM-006", "receipt_no": "REC-1006", "member_id": "MEM-GYM-1006", "member_name": "Sara Khan", "plan": "Quarterly", "paid_amount": 0.00, "total_amount": 2100.00, "payment_method": "Pending", "payment_date": "2026-05-01", "status": "Overdue"},
    {"payment_id": "PAY-GYM-007", "receipt_no": "REC-1007", "member_id": "MEM-GYM-1007", "member_name": "Vishnu Raj", "plan": "Monthly", "paid_amount": 1000.00, "total_amount": 1000.00, "payment_method": "UPI", "payment_date": "2026-08-05", "status": "Paid"},
    {"payment_id": "PAY-GYM-008", "receipt_no": "REC-1008", "member_id": "MEM-GYM-1008", "member_name": "Anjali Menon", "plan": "New Year", "paid_amount": 2000.00, "total_amount": 2500.00, "payment_method": "Bank Transfer", "payment_date": "2026-02-22", "status": "Partial"},
])

fitness_centers = pd.DataFrame([
    {"center_id": "CTR-001", "center_name": "Downtown Fitness Club", "city": "Kochi", "address": "MG Road, Ernakulam", "capacity": 250, "manager": "David Miller", "status": "Active"},
    {"center_id": "CTR-002", "center_name": "Uptown Health Hub", "city": "Kakkanad", "address": "Infopark Road, Kakkanad", "capacity": 180, "manager": "Priya Patel", "status": "Active"},
    {"center_id": "CTR-003", "center_name": "Beachside Fitness Center", "city": "Kozhikode", "address": "Beach Road, Kozhikode", "capacity": 140, "manager": "Manu Varghese", "status": "Maintenance"},
])

equipment_suppliers = pd.DataFrame([
    {"supplier_id": "SUP-GYM-001", "supplier_company": "Rogue Fitness Machinery", "contact_person": "Alex Mercer", "email": "support@roguefitness.com", "phone": "+1 800 555 0199", "country": "USA", "category": "Gym Machinery", "status": "Active"},
    {"supplier_id": "SUP-GYM-002", "supplier_company": "Optimum Nutrition Supplies", "contact_person": "Sarah Jenkins", "email": "b2b@optimumdist.com", "phone": "+1 800 555 0244", "country": "USA", "category": "Nutrition", "status": "Active"},
    {"supplier_id": "SUP-GYM-003", "supplier_company": "Kerala Fitness Equipments", "contact_person": "Shyam Kumar", "email": "sales@kfe.in", "phone": "+91 98765 40003", "country": "India", "category": "Weights", "status": "Active"},
    {"supplier_id": "SUP-GYM-004", "supplier_company": "Flex Flooring Systems", "contact_person": "Nisha Roy", "email": "orders@flexfloor.in", "phone": "+91 98765 40004", "country": "India", "category": "Flooring", "status": "Active"},
    {"supplier_id": "SUP-GYM-005", "supplier_company": "Cardio Tech Solutions", "contact_person": "Imran Ali", "email": "help@cardiotech.in", "phone": "+91 98765 40005", "country": "India", "category": "Cardio Machines", "status": "Inactive"},
])

expenses_accounts = pd.DataFrame([
    {"expense_id": "EXP-001", "expense_date": "2026-08-01", "category": "Rent", "description": "Downtown branch monthly rent", "amount": 75000.00, "payment_method": "Bank Transfer", "status": "Paid"},
    {"expense_id": "EXP-002", "expense_date": "2026-08-03", "category": "Electricity", "description": "Electricity bill", "amount": 18500.00, "payment_method": "UPI", "status": "Paid"},
    {"expense_id": "EXP-003", "expense_date": "2026-08-05", "category": "Equipment", "description": "New dumbbell set", "amount": 42000.00, "payment_method": "Bank Transfer", "status": "Paid"},
    {"expense_id": "EXP-004", "expense_date": "2026-08-08", "category": "Maintenance", "description": "Treadmill service", "amount": 12500.00, "payment_method": "Cash", "status": "Paid"},
    {"expense_id": "EXP-005", "expense_date": "2026-08-10", "category": "Salary", "description": "Trainer salary advance", "amount": 30000.00, "payment_method": "Bank Transfer", "status": "Paid"},
    {"expense_id": "EXP-006", "expense_date": "2026-08-15", "category": "Marketing", "description": "Social media campaign", "amount": 15000.00, "payment_method": "Credit Card", "status": "Pending"},
    {"expense_id": "EXP-007", "expense_date": "2026-08-18", "category": "Cleaning", "description": "Cleaning supplies", "amount": 6500.00, "payment_method": "UPI", "status": "Paid"},
    {"expense_id": "EXP-008", "expense_date": "2026-08-20", "category": "Nutrition", "description": "Protein sample stock", "amount": 22000.00, "payment_method": "Bank Transfer", "status": "Pending"},
])

employees_staff = pd.DataFrame([
    {"employee_id": "EMP-GYM-01", "employee_name": "Marcus Vance", "role": "Senior Personal Trainer", "email": "marcus.trainer@gymfitness.com", "phone": "+91 98765 11111", "branch": "Downtown Fitness Club", "status": "Active"},
    {"employee_id": "EMP-GYM-02", "employee_name": "Priya Patel", "role": "Sports Nutritionist", "email": "priya.nutrition@gymfitness.com", "phone": "+91 98765 22222", "branch": "Uptown Health Hub", "status": "Active"},
    {"employee_id": "EMP-GYM-03", "employee_name": "David Miller", "role": "Desk & Check-in Specialist", "email": "david.desk@gymfitness.com", "phone": "+91 98765 33333", "branch": "Downtown Fitness Club", "status": "Active"},
    {"employee_id": "EMP-GYM-04", "employee_name": "Fahis", "role": "Bodybuilding Trainer", "email": "fahis@gymfitness.com", "phone": "+91 98765 44444", "branch": "Downtown Fitness Club", "status": "Active"},
    {"employee_id": "EMP-GYM-05", "employee_name": "Anu Thomas", "role": "Accounts Executive", "email": "anu.accounts@gymfitness.com", "phone": "+91 98765 55555", "branch": "Uptown Health Hub", "status": "Active"},
    {"employee_id": "EMP-GYM-06", "employee_name": "Rakesh Babu", "role": "Maintenance Technician", "email": "rakesh.maintenance@gymfitness.com", "phone": "+91 98765 66666", "branch": "Beachside Fitness Center", "status": "Inactive"},
])

system_settings = pd.DataFrame([
    {"setting_section": "Business Profile", "configuration": "Company identity, logo, tax/VAT, email, phone and headquarters address", "current_value": "Company: Energy Gym"},
    {"setting_section": "Localization & Finance", "configuration": "Currency, currency symbol, timezone, date format and fiscal year", "current_value": "USD ($), UTC, YYYY-MM-DD, January"},
    {"setting_section": "Invoicing & Sales", "configuration": "Invoice, sales order, purchase order prefixes and tax rate", "current_value": "INV-, SO-, PO-, Tax 0%"},
    {"setting_section": "Inventory & POS", "configuration": "Low-stock alerts, threshold and POS receipt options", "current_value": "Alerts enabled, threshold 10, 80mm thermal"},
    {"setting_section": "Warehouse & Stock", "configuration": "Warehouse tracking, stock alerts and transfer actions", "current_value": "Multi-warehouse disabled, alerts enabled"},
    {"setting_section": "Barcode Printing", "configuration": "Barcode generator, paper size and encoding standard", "current_value": "80mm thermal, CODE128"},
    {"setting_section": "Landing Page Settings", "configuration": "Navbar, hero banner, about section and footer content", "current_value": "ERP Cloud landing content"},
    {"setting_section": "Security & System", "configuration": "Email notifications, daily summary, session timeout and theme", "current_value": "Notifications enabled, timeout 60 minutes"},
])

reports_analytics = pd.DataFrame([
    {"metric": "Total Members", "value": len(gym_members), "period": "Current", "category": "Memberships"},
    {"metric": "Active Members", "value": int((gym_members["status"] == "Active").sum()), "period": "Current", "category": "Memberships"},
    {"metric": "Inactive / Expired Members", "value": int((gym_members["status"] != "Active").sum()), "period": "Current", "category": "Memberships"},
    {"metric": "Collected Revenue", "value": float(payments_fees["paid_amount"].sum()), "period": "2026", "category": "Payments & Revenue"},
    {"metric": "Outstanding Fees", "value": float((payments_fees["total_amount"] - payments_fees["paid_amount"]).sum()), "period": "2026", "category": "Payments & Revenue"},
    {"metric": "Present Check-ins", "value": int((attendance_log["status"] == "Present").sum()), "period": "Dataset", "category": "Attendance"},
])


# ============================================================
# ERP MODULES
# ============================================================

ERP_PATHS = {
    "Dashboard": "ERP → Gym Management → Dashboard",
    "Gym Members": "ERP → Gym Management → Gym Members",
    "Membership Plans": "ERP → Gym Management → Membership Plans",
    "Gym Trainers": "ERP → Gym Management → Gym Trainers",
    "Attendance Log": "ERP → Gym Management → Attendance Log",
    "Payments & Fees": "ERP → Gym Management → Payments & Fees",
    "Fitness Centers & Facilities": "ERP → Gym Management → Fitness Centers & Facilities",
    "Equipment Suppliers": "ERP → Gym Management → Equipment Suppliers",
    "Expenses & Accounts": "ERP → Gym Management → Expenses & Accounts",
    "Employees / Staff": "ERP → Gym Management → Employees / Staff",
    "Reports & Analytics": "ERP → Gym Management → Reports & Analytics",
    "Business Profile": "ERP → Gym Management → Settings → Business Profile",
    "Localization & Finance": "ERP → Gym Management → Settings → Localization & Finance",
    "Invoicing & Sales": "ERP → Gym Management → Settings → Invoicing & Sales",
    "Inventory & POS": "ERP → Gym Management → Settings → Inventory & POS",
    "Warehouse & Stock": "ERP → Gym Management → Settings → Warehouse & Stock",
    "Barcode Printing": "ERP → Gym Management → Settings → Barcode Printing",
    "Landing Page Settings": "ERP → Gym Management → Settings → Landing Page Settings",
    "Security & System": "ERP → Gym Management → Settings → Security & System",
}

DATASET_REGISTRY = {
    "Gym Members": gym_members,
    "Membership Plans": membership_plans,
    "Gym Trainers": gym_trainers,
    "Attendance Log": attendance_log,
    "Payments & Fees": payments_fees,
    "Fitness Centers & Facilities": fitness_centers,
    "Equipment Suppliers": equipment_suppliers,
    "Expenses & Accounts": expenses_accounts,
    "Employees / Staff": employees_staff,
    "Reports & Analytics": reports_analytics,
}

MODULE_ALIASES = {
    "gym member": "Gym Members",
    "members": "Gym Members",
    "membership plan": "Membership Plans",
    "plan price": "Membership Plans",
    "gym trainer": "Gym Trainers",
    "trainer": "Gym Trainers",
    "attendance analytics": "Reports & Analytics",
    "attendance report": "Reports & Analytics",
    "attendance": "Attendance Log",
    "check in": "Attendance Log",
    "check out": "Attendance Log",
    "payment": "Payments & Fees",
    "membership fee": "Payments & Fees",
    "receipt": "Payments & Fees",
    "fitness center": "Fitness Centers & Facilities",
    "facility": "Fitness Centers & Facilities",
    "gym branch": "Fitness Centers & Facilities",
    "supplier": "Equipment Suppliers",
    "equipment vendor": "Equipment Suppliers",
    "expense": "Expenses & Accounts",
    "accounts": "Expenses & Accounts",
    "employee": "Employees / Staff",
    "staff": "Employees / Staff",
    "report": "Reports & Analytics",
    "analytics": "Reports & Analytics",
    "business profile": "Business Profile",
    "company logo": "Business Profile",
    "localization": "Localization & Finance",
    "currency": "Localization & Finance",
    "timezone": "Localization & Finance",
    "invoicing": "Invoicing & Sales",
    "sales prefix": "Invoicing & Sales",
    "inventory pos": "Inventory & POS",
    "low stock": "Inventory & POS",
    "warehouse": "Warehouse & Stock",
    "stock transfer": "Warehouse & Stock",
    "barcode": "Barcode Printing",
    "landing page": "Landing Page Settings",
    "security": "Security & System",
    "session timeout": "Security & System",
}

TRAINING_QUERIES = [
    ("show dashboard", "Dashboard"),
    ("open gym overview", "Dashboard"),
    ("show all gym members", "Gym Members"),
    ("find active members", "Gym Members"),
    ("search member by name", "Gym Members"),
    ("show membership plans", "Membership Plans"),
    ("annual plan price", "Membership Plans"),
    ("monthly gym package", "Membership Plans"),
    ("show gym trainers", "Gym Trainers"),
    ("find bodybuilding coach", "Gym Trainers"),
    ("trainer experience", "Gym Trainers"),
    ("member attendance log", "Attendance Log"),
    ("today check in details", "Attendance Log"),
    ("show absent members", "Attendance Log"),
    ("show gym payments", "Payments & Fees"),
    ("find overdue membership fees", "Payments & Fees"),
    ("credit card payment receipts", "Payments & Fees"),
    ("show fitness centers", "Fitness Centers & Facilities"),
    ("gym branches and facilities", "Fitness Centers & Facilities"),
    ("show equipment suppliers", "Equipment Suppliers"),
    ("nutrition vendors", "Equipment Suppliers"),
    ("show gym expenses", "Expenses & Accounts"),
    ("pending accounts expenses", "Expenses & Accounts"),
    ("show employees and staff", "Employees / Staff"),
    ("staff roster", "Employees / Staff"),
    ("membership analytics", "Reports & Analytics"),
    ("revenue report", "Reports & Analytics"),
    ("business profile settings", "Business Profile"),
    ("company logo and address", "Business Profile"),
    ("currency and timezone settings", "Localization & Finance"),
    ("financial localization", "Localization & Finance"),
    ("invoice sales prefix settings", "Invoicing & Sales"),
    ("purchase order tax configuration", "Invoicing & Sales"),
    ("inventory POS low stock", "Inventory & POS"),
    ("receipt printer settings", "Inventory & POS"),
    ("warehouse stock transfer", "Warehouse & Stock"),
    ("multi location stock", "Warehouse & Stock"),
    ("barcode printing settings", "Barcode Printing"),
    ("open barcode generator", "Barcode Printing"),
    ("landing page branding", "Landing Page Settings"),
    ("hero banner footer content", "Landing Page Settings"),
    ("security notifications", "Security & System"),
    ("session timeout theme", "Security & System"),
]

training_df = pd.DataFrame(TRAINING_QUERIES, columns=["query", "module"])


# ============================================================
# SEARCH ENGINE
# ============================================================

def clean_text(text):
    text = str(text).lower().replace("&", " and ").replace("/", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def build_intent_model():
    cleaned = training_df["query"].map(clean_text)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english"
    )
    vectors = vectorizer.fit_transform(cleaned)
    return vectorizer, vectors


def detect_module(user_query):
    query = clean_text(user_query)

    for alias, module in sorted(
        MODULE_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True
    ):
        if alias in query:
            return module, 1.0

    vectorizer, vectors = build_intent_model()
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, vectors)[0]

    best_index = int(np.argmax(similarities))
    return (
        training_df.iloc[best_index]["module"],
        round(float(similarities[best_index]), 4)
    )


def apply_metadata_filters(data, user_query):
    result = data.copy()
    query = clean_text(user_query)

    if "status" in result.columns:
        status_rules = {
            "active": "Active",
            "inactive": "Inactive",
            "expired": "Expired",
            "overdue": "Overdue",
            "partial": "Partial",
            "pending": "Pending",
            "paid": "Paid",
            "present": "Present",
            "absent": "Absent",
        }

        for keyword, value in status_rules.items():
            if keyword in query:
                filtered = result[
                    result["status"].astype(str).str.lower() == value.lower()
                ]
                if not filtered.empty:
                    result = filtered
                break

    if "payment_method" in result.columns:
        method_rules = [
            "credit card",
            "debit card",
            "upi",
            "cash",
            "bank transfer",
        ]

        for method in method_rules:
            if method in query:
                filtered = result[
                    result["payment_method"].astype(str).str.lower() == method
                ]
                if not filtered.empty:
                    result = filtered
                break

    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", query)

    if date_match:
        date_value = date_match.group(0)
        date_columns = [
            column for column in result.columns
            if "date" in column
        ]

        for column in date_columns:
            filtered = result[
                result[column].astype(str) == date_value
            ]

            if not filtered.empty:
                result = filtered
                break

    amount_match = re.search(
        r"(?:above|over|greater than|below|under|less than)\s+(\d+(?:\.\d+)?)",
        query
    )

    if amount_match:
        amount = float(amount_match.group(1))

        amount_columns = [
            column
            for column in [
                "paid_amount",
                "total_amount",
                "amount",
                "price",
                "value",
            ]
            if column in result.columns
        ]

        if amount_columns:
            column = amount_columns[0]

            if any(
                word in query
                for word in ["above", "over", "greater than"]
            ):
                result = result[
                    pd.to_numeric(
                        result[column],
                        errors="coerce"
                    ) > amount
                ]
            else:
                result = result[
                    pd.to_numeric(
                        result[column],
                        errors="coerce"
                    ) < amount
                ]

    return result.reset_index(drop=True)


def rank_records(data, user_query):
    if data.empty:
        return data.assign(
            relevance_score=pd.Series(dtype=float)
        )

    record_text = (
        data.fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .map(clean_text)
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(
        record_text.tolist() + [clean_text(user_query)]
    )

    scores = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    )[0]

    ranked = data.copy()
    ranked["relevance_score"] = scores.round(4)

    return ranked.sort_values(
        "relevance_score",
        ascending=False
    ).reset_index(drop=True)


def smart_erp_search(user_query, top_k=5):
    module, intent_score = detect_module(user_query)

    exact_path = ERP_PATHS[module]

    if module in DATASET_REGISTRY:
        data = DATASET_REGISTRY[module].copy()
    elif module in system_settings["setting_section"].values:
        data = system_settings[
            system_settings["setting_section"] == module
        ].copy()
    else:
        return {
            "module": module,
            "score": intent_score,
            "path": exact_path,
            "data": pd.DataFrame()
        }

    filtered = apply_metadata_filters(data, user_query)
    ranked = rank_records(filtered, user_query)

    return {
        "module": module,
        "score": intent_score,
        "path": exact_path,
        "data": ranked.head(top_k)
    }


# ============================================================
# DASHBOARD
# ============================================================

def dashboard():
    st.markdown(
        '<div class="main-title">🏋️ Gym ERP AI Smart Search</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-powered natural language search for your Gym ERP system</div>',
        unsafe_allow_html=True
    )

    total_members = len(gym_members)
    active_members = int(
        (gym_members["status"] == "Active").sum()
    )
    revenue = float(
        payments_fees["paid_amount"].sum()
    )
    outstanding = float(
        (
            payments_fees["total_amount"]
            - payments_fees["paid_amount"]
        ).sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("👥 Total Members", total_members)

    with c2:
        st.metric("✅ Active Members", active_members)

    with c3:
        st.metric("💰 Revenue", f"₹{revenue:,.0f}")

    with c4:
        st.metric("⚠️ Outstanding", f"₹{outstanding:,.0f}")

    st.divider()

    st.subheader("🔎 AI Search")

    query = st.text_input(
        "Ask your Gym ERP a question",
        placeholder="Example: show active gym members",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([5, 1])

    with col1:
        st.caption(
            "Try: active members • overdue fees • credit card payments • "
            "bodybuilding trainer • revenue report"
        )

    with col2:
        search_clicked = st.button(
            "🔍 Search",
            use_container_width=True,
            type="primary"
        )

    if search_clicked and query.strip():
        result = smart_erp_search(query.strip(), top_k=10)

        st.session_state["last_result"] = result
        st.session_state["last_query"] = query.strip()

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]

        st.subheader("🎯 Search Result")

        m1, m2 = st.columns(2)

        with m1:
            st.info(
                f"**Detected Module:** {result['module']}\n\n"
                f"**Confidence:** {result['score'] * 100:.1f}%"
            )

        with m2:
            st.success(
                f"**ERP Path:** {result['path']}"
            )

        data = result["data"]

        if data.empty:
            st.warning("No matching records found.")
        else:
            st.caption(
                f"Showing {len(data)} matching record(s)"
            )

            display_data = data.copy()

            if "relevance_score" in display_data.columns:
                display_data["relevance_score"] = (
                    display_data["relevance_score"]
                    .map(lambda x: f"{x:.4f}")
                )

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🏋️ Gym ERP")

st.sidebar.caption("AI Management System")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "👥 Gym Members",
        "💳 Payments & Fees",
        "🏋️ Gym Trainers",
        "📅 Attendance Log",
        "📋 Membership Plans",
        "🏢 Fitness Centers",
        "🚚 Equipment Suppliers",
        "💰 Expenses & Accounts",
        "👨‍💼 Employees / Staff",
        "📊 Reports & Analytics",
        "⚙️ Settings",
    ],
)

st.sidebar.divider()
st.sidebar.caption("AI Engine")
st.sidebar.success("TF-IDF + Cosine Similarity")


# ============================================================
# PAGE CONTENT
# ============================================================

if menu == "🏠 Dashboard":
    dashboard()

elif menu == "👥 Gym Members":
    st.title("👥 Gym Members")
    st.dataframe(gym_members, use_container_width=True, hide_index=True)

elif menu == "💳 Payments & Fees":
    st.title("💳 Payments & Fees")
    st.dataframe(payments_fees, use_container_width=True, hide_index=True)

elif menu == "🏋️ Gym Trainers":
    st.title("🏋️ Gym Trainers")
    st.dataframe(gym_trainers, use_container_width=True, hide_index=True)

elif menu == "📅 Attendance Log":
    st.title("📅 Attendance Log")
    st.dataframe(attendance_log, use_container_width=True, hide_index=True)

elif menu == "📋 Membership Plans":
    st.title("📋 Membership Plans")
    st.dataframe(membership_plans, use_container_width=True, hide_index=True)

elif menu == "🏢 Fitness Centers":
    st.title("🏢 Fitness Centers & Facilities")
    st.dataframe(fitness_centers, use_container_width=True, hide_index=True)

elif menu == "🚚 Equipment Suppliers":
    st.title("🚚 Equipment Suppliers")
    st.dataframe(equipment_suppliers, use_container_width=True, hide_index=True)

elif menu == "💰 Expenses & Accounts":
    st.title("💰 Expenses & Accounts")
    st.dataframe(expenses_accounts, use_container_width=True, hide_index=True)

elif menu == "👨‍💼 Employees / Staff":
    st.title("👨‍💼 Employees / Staff")
    st.dataframe(employees_staff, use_container_width=True, hide_index=True)

elif menu == "📊 Reports & Analytics":
    st.title("📊 Reports & Analytics")
    st.dataframe(reports_analytics, use_container_width=True, hide_index=True)

elif menu == "⚙️ Settings":
    st.title("⚙️ Settings")

    setting_names = system_settings["setting_section"].tolist()

    selected = st.selectbox(
        "Select Settings Module",
        setting_names
    )

    selected_data = system_settings[
        system_settings["setting_section"] == selected
    ]

    st.dataframe(
        selected_data,
        use_container_width=True,
        hide_index=True
    )

st.sidebar.divider()
st.sidebar.caption("Gym ERP AI Smart Search • Demo Dataset")
