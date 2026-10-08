"""
classifier.py — Transaction category classifier

Two-stage pipeline:
  Stage 1: Heuristic rule engine
      • ~40 hand-crafted regex rules covering Indian banking narrations.
      • Rules are direction-aware (credit vs. debit).
      • First match wins (priority order).

  Stage 2: TF-IDF + Logistic Regression (fallback)
      • Character-level n-gram TF-IDF vectoriser (handles noisy, abbreviated text).
      • Trained on user-feedback + seed data loaded from data/seed_labels.csv.
      • Triggered only when no rule matches and confidence ≥ min_conf.

Incremental learning:
      • learn() accumulates labelled rows to data/feedback.csv and retrains.
      • User corrections weighted 3× to quickly override rules.
"""

from __future__ import annotations
import logging
import os
import re

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

log = logging.getLogger(__name__)

# Rule table 
# (category, regex_pattern, required_direction)
# direction: "dr" | "cr" | None (any)

_RULES_RAW = [
    #  Income 
    ("Salary",                r"salary|payroll|stipend|\bsal\s*cr\b|emolument",                              "cr"),
    ("Interest Income",       r"\bint\.?\s*(pd|paid|credit|cr)\b|interest\s*credited|savings\s*int",         "cr"),
    ("Dividend",              r"dividend|div\s*paid|nsdl.*div|cdsl.*div",                                    "cr"),
    ("Refund / Reversal",     r"refund|reversal|\brev\b|cashback|chargeback|reimburs",                        "cr"),

    #  Cash 
    ("Cash Withdrawal (ATM)", r"\batm\b|\bnwd\b|cash\s*(wdl|withdrawal)|\bcwdr\b|atm\s*cash",                "dr"),
    ("Cash Deposit",          r"cash\s*dep|\bcdm\b|by\s*cash|cash\s*deposi",                                 "cr"),

    #  Bank Charges 
    ("Bank Charges & Fees",   r"charges?\b|\bchrg|\bfee\b|igst|cgst|sgst|sms\s*alert|\bamc\b"
                              r"|min(imum)?\s*bal|penalty|annual\s*fee|service\s*charge"
                              r"|locker\s*charge|processing\s*fee|late\s*payment",                           "dr"),

    #  Tax & Govt 
    ("Tax",                   r"income\s*tax|\bitd\b|\btds\b|cbdt|advance\s*tax|gst\s*payment"
                              r"|tds\s*deducted|challan",                                                     None),

    #  Loans & EMI 
    ("EMI / Loan",            r"\bemi\b|loan|repayment|bajaj\s*fin|\bnach\b.*\bfin"
                              r"|equit[ae]d|principal|lic\s*housing|pnb\s*housing"
                              r"|home\s*loan|vehicle\s*loan|personal\s*loan",                                 "dr"),

    #  Insurance 
    ("Insurance",             r"\blic\b|insurance|premium|policybazaar|star\s*health"
                              r"|hdfc\s*life|icici\s*pru|max\s*life|sbi\s*life"
                              r"|tata\s*aia|care\s*health|niva\s*bupa",                                       "dr"),

    #  Investments 
    ("Investments",           r"zerodha|groww|upstox|\bsip\b|mutual\s*fund|\bnps\b|\bppf\b"
                              r"|kuvera|indmoney|\bnsdl\b|\bcdsl\b|smallcase|coin\s*by"
                              r"|demat|nifty\s*bees|ipo\s*allot|subscription\s*fee.*fund",                   None),

    #  Food & Dining 
    ("Food & Dining",         r"swiggy|zomato|domino|mcdonald|\bkfc\b|restaurant|cafe"
                              r"|starbucks|pizza|eatsure|burger|dunkin|subway|haldiram"
                              r"|barbeque|biryani|fresh\s*menu|box8|faasos|inner\s*chef",                    "dr"),

    #  Groceries 
    ("Groceries",             r"bigbasket|blinkit|zepto|dmart|grofers|instamart"
                              r"|reliance\s*fresh|grocer|supermarket|\bmilk\b"
                              r"|nature\s*basket|more\s*retail|lulu|spar",                                   "dr"),

    #  Shopping 
    ("Shopping",              r"amazon|flipkart|myntra|ajio|meesho|nykaa|decathlon"
                              r"|ikea|croma|reliance\s*digital|snapdeal|paytm\s*mall"
                              r"|tata\s*cliq|shopclues|firstcry|babyoye|lifestyle"
                              r"|pantaloon|westside|max\s*fashion|\bshopify\b",                               "dr"),

    #  Transport & Fuel 
    ("Transport & Fuel",      r"\buber\b|\bola\b|rapido|fuel|petrol|diesel|\bhpcl\b"
                              r"|\bbpcl\b|\biocl\b|indian\s*oil|fastag|metro\s*rail"
                              r"|rapido|\bnamma\s*metro\b|auto\s*rickshaw|cab|parking",                       "dr"),

    #  Travel 
    ("Travel",                r"irctc|makemytrip|goibibo|indigo|air\s*india|cleartrip"
                              r"|redbus|ixigo|\boyo\b|airbnb|yatra|abhibus|spicejet"
                              r"|vistara|akasa|easemytrip|hotel|resort|booking\.com",                         "dr"),

    #  Utilities & Bills 
    ("Utilities & Bills",     r"electric|bescom|mseb|torrent\s*power|tata\s*power"
                              r"|water\s*bill|piped\s*gas|billdesk|bbps|bill\s*pay"
                              r"|mahanagar\s*gas|igl\b|mgl\b|adani\s*gas|cesc\b"
                              r"|tneb\b|kseb\b|wesco\b|nesco\b|reliance\s*energy",                           "dr"),

    #  Mobile & Internet 
    ("Mobile & Internet",     r"airtel|jio\b|vodafone|\bvi\b|bsnl|recharge|broadband"
                              r"|fibernet|hathway|tata\s*play|act\s*fibernet|den\s*networks"
                              r"|you\s*broadband|excitel|beam\s*fiber",                                       "dr"),

    #  Entertainment 
    ("Entertainment",         r"netflix|spotify|hotstar|prime\s*video|bookmyshow|\bpvr\b"
                              r"|\binox\b|youtube|sony\s*liv|zee5|lionsgate|apple\s*tv"
                              r"|disney\s*plus|mxplayer|jio\s*cinema|aha\b|voot",                            "dr"),

    #  Healthcare 
    ("Healthcare",            r"pharmacy|apollo|medplus|hospital|clinic|\b1mg\b"
                              r"|pharmeasy|diagnostic|netmeds|practo|lal\s*path"
                              r"|thyrocare|fortis|max\s*hospital|manipal|aiims",                              "dr"),

    #  Education 
    ("Education",             r"school|college|universit|tuition|udemy|coursera"
                              r"|exam\s*fee|byju|unacademy|vedantu|simplilearn"
                              r"|khan\s*academy|upgrad|whitehat|coding\s*ninjas",                             "dr"),

    #  Rent 
    ("Rent",                  r"\brent\b|landlord|nobroker|magicbricks|99acres|housing\.com"
                              r"|maintenance\s*charge|society\s*charge",                                      "dr"),

    #  PayTM / Wallets 
    ("Wallet / UPI",          r"paytm|phonepe|googlepay|\bgpay\b|amazon\s*pay|mobikwik"
                              r"|freecharge|cred\b|slice\b|jupiter\b|\bfi\b\s*money"
                              r"|navi\b|bhim\b",                                                              None),

    #  Self Transfer 
    ("Self Transfer",         r"\bself\b|own\s*account|to\s*self|a/?c\s*to\s*a/?c",                         None),

    #  Cheque 
    ("Cheque",                r"\bchq\b|cheque|\bclg\b|clearing\b|instrument",                               None),

    #  Generic Transfers 
    ("Transfers (Person)",    r"\bupi\b|\bneft\b|\bimps\b|\brtgs\b",                                        None),
]

RULES = [(cat, re.compile(pat, re.I), dirn) for cat, pat, dirn in _RULES_RAW]
CATEGORIES = [cat for cat, _, _ in _RULES_RAW] + ["Others"]

#Noise tokens for counterparty extraction
_NOISE_RE = re.compile(
    r"^(upi|neft|imps|rtgs|ach|nach|pos|ecom|payment|from|to|dr|cr|ref|trf"
    r"|transfer|ybl|okaxis|oksbi|okicici|paytm|ibl|axl|sbi|hdfc|icici|axis"
    r"|kotak|pnb|canara|union|federal|yes|indus|rbl|au|idfc|fino|airtel)$",
    re.I,
)


#Utility functions

def counterparty(desc: str | None) -> str:
    """
    Heuristically extract the payee / remitter from a UPI/NEFT narration.
    Handles patterns like:
        UPI/12345/SWIGGY ORDER/...
        NEFT-HDFC-JOHN DOE-REF12345
        POS 12345 AMAZON
    """
    for part in re.split(r"[/\-@|]", desc or ""):
        part = part.strip()
        if (
            len(part) >= 3
            and re.search(r"[A-Za-z]{3,}", part)
            and not _NOISE_RE.fullmatch(part)
            and not re.search(r"\d{5,}", part)
        ):
            return part.title()[:40]
    return ""


def _featurise(desc: str | None, direction: str) -> str:
    """
    Clean and featurise a description for ML.
    • Lowercase, remove punctuation
    • Remove long digit sequences (reference numbers)
    • Append direction token __dr__ / __cr__
    """
    text = re.sub(r"[^a-z0-9 ]", " ", (desc or "").lower())
    text = " ".join(w for w in text.split() if not (w.isdigit() and len(w) >= 4))
    return f"{text} __{direction}__"


def _direction(debit, credit) -> str:
    return "dr" if (pd.notna(debit) and debit) else "cr"


#Classifier class

class Classifier:
    """
    Two-stage transaction classifier:
      1. Rule engine (regex, direction-aware)
      2. TF-IDF + Logistic Regression (when rules don't match)
    """

    def __init__(
        self,
        model_path:    str   = "data/model.joblib",
        feedback_path: str   = "data/feedback.csv",
        seed_path:     str   = "data/seed_labels.csv",
        min_conf:      float = 0.50,
    ):
        self.model_path    = model_path
        self.feedback_path = feedback_path
        self.seed_path     = seed_path
        self.min_conf      = min_conf
        self.model         = None

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Load persisted model if available
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                log.info("Loaded ML model from %s", model_path)
            except Exception as exc:
                log.warning("Could not load model: %s", exc)

        # Bootstrap from seed data if model is missing
        if self.model is None and os.path.exists(seed_path):
            log.info("Bootstrapping classifier from seed data …")
            seed = pd.read_csv(seed_path)
            if {"Description", "Direction", "Category"}.issubset(seed.columns):
                seed["Source"] = "seed"
                self._train(seed)

    # Single transaction

    def classify(
        self,
        desc:   str | None,
        debit,
        credit,
    ) -> tuple[str, float, str]:
        """
        Classify a single transaction.

        Returns:
            (category, confidence, method)
            method: "rule" | "ml" | "default"
        """
        dirn = _direction(debit, credit)

        # Stage 1 — rules
        for cat, rx, required_dir in RULES:
            if (required_dir is None or required_dir == dirn) and rx.search(desc or ""):
                return cat, 0.95, "rule"

        # Stage 2 — ML
        if self.model is not None:
            feat = _featurise(desc, dirn)
            try:
                proba = self.model.predict_proba([feat])[0]
                best_idx = proba.argmax()
                conf = float(proba[best_idx])
                if conf >= self.min_conf:
                    return self.model.classes_[best_idx], conf, "ml"
            except Exception as exc:
                log.warning("ML inference error: %s", exc)

        return "Others", 0.0, "default"

    # DataFrame batch classify 

    def classify_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Category, Confidence, Method columns to a transactions DataFrame.
        """
        results = [
            self.classify(row.Description, row.Debit, row.Credit)
            for row in df.itertuples(index=False)
        ]
        if results:
            cats, confs, methods = zip(*results)
        else:
            cats, confs, methods = [], [], []

        df = df.copy()
        df["Category"]   = list(cats)
        df["Confidence"] = list(confs)
        df["Method"]     = list(methods)
        return df

    #Incremental learning 

    def learn(self, rows: pd.DataFrame) -> int:
        """
        Accept new labelled rows (Description, Direction, Category, Source)
        and retrain the ML model.

        User corrections (Source='user') are weighted 3× over rules/seed.
        Returns the total number of training rows used.
        """
        required = {"Description", "Direction", "Category"}
        if not required.issubset(rows.columns):
            raise ValueError(f"rows must have columns: {required}")

        # Merge with existing feedback
        old = (
            pd.read_csv(self.feedback_path)
            if os.path.exists(self.feedback_path)
            else pd.DataFrame(columns=rows.columns)
        )
        combined = (
            pd.concat([old, rows], ignore_index=True)
            .drop_duplicates(["Description", "Direction", "Category"], keep="last")
        )
        combined.to_csv(self.feedback_path, index=False)

        n = self._train(combined)
        return n


    def _train(self, data: pd.DataFrame) -> int:
        """Train TF-IDF + LR on data; persist model. Returns training row count."""
        if len(data) < 20 or data["Category"].nunique() < 2:
            log.info("Not enough data to train (%d rows, %d cats)", len(data), data["Category"].nunique())
            return len(data)

        # Weight user corrections 3×, seed 1×
        source = data.get("Source", pd.Series(["seed"] * len(data)))
        weight_map = {"user": 3, "rule": 1, "seed": 1, "default": 1}
        weights = source.map(weight_map).fillna(1).astype(int)
        train = pd.concat(
            [data] + [data[source == "user"]] * 2,
            ignore_index=True,
        )

        X = [_featurise(d, dr) for d, dr in zip(train["Description"], train["Direction"])]
        y = train["Category"].tolist()

        pipeline = make_pipeline(
            TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=50_000,
            ),
            # LR model to make it explicitly calibrated
            CalibratedClassifierCV(
                estimator=LogisticRegression(max_iter=2000, C=10, class_weight="balanced", solver="lbfgs"),
                method='sigmoid',
                cv=3
            ),
        )
        pipeline.fit(X, y)
        self.model = pipeline
        joblib.dump(pipeline, self.model_path)
        log.info("Model trained on %d rows, saved to %s", len(train), self.model_path)
        return len(data)