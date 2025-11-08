"""
Feature engineering utilities for loan default prediction
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


class FeatureEngineer:
    """Handles feature engineering for loan default prediction"""

    def __init__(self):
        """Initialize feature engineer"""
        self.feature_stats = {}

    def create_loan_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create loan-specific features

        Args:
            df: Input dataframe

        Returns:
            DataFrame with additional loan features
        """
        df = df.copy()

        # Loan to income ratio
        if 'loan_amount' in df.columns and 'annual_income' in df.columns:
            df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income'].replace(0, np.nan)

        # Payment burden (monthly payment as % of monthly income)
        if 'installment' in df.columns and 'annual_income' in df.columns:
            df['payment_burden_ratio'] = (df['installment'] * 12) / df['annual_income'].replace(0, np.nan)

        # Funded vs requested amount
        if 'funded_amount' in df.columns and 'loan_amount' in df.columns:
            df['funded_ratio'] = df['funded_amount'] / df['loan_amount'].replace(0, np.nan)

        return df

    def create_credit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create credit-related features

        Args:
            df: Input dataframe

        Returns:
            DataFrame with additional credit features
        """
        df = df.copy()

        # FICO average
        if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
            df['fico_score_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2

        # Credit history length
        if 'earliest_credit_line' in df.columns and 'issue_date' in df.columns:
            df['earliest_credit_line'] = pd.to_datetime(df['earliest_credit_line'], errors='coerce')
            df['issue_date'] = pd.to_datetime(df['issue_date'], errors='coerce')
            df['credit_history_years'] = (
                (df['issue_date'] - df['earliest_credit_line']).dt.days / 365.25
            )

        # Account utilization
        if 'open_accounts' in df.columns and 'total_accounts' in df.columns:
            df['account_open_ratio'] = df['open_accounts'] / df['total_accounts'].replace(0, np.nan)

        # Recent account activity
        if 'accounts_opened_past_24months' in df.columns and 'total_accounts' in df.columns:
            df['recent_account_ratio'] = (
                df['accounts_opened_past_24months'] / df['total_accounts'].replace(0, np.nan)
            )

        return df

    def create_delinquency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create delinquency-related features

        Args:
            df: Input dataframe

        Returns:
            DataFrame with additional delinquency features
        """
        df = df.copy()

        # Has any delinquency
        if 'delinquencies_2yrs' in df.columns:
            df['has_delinquency'] = (df['delinquencies_2yrs'] > 0).astype(int)

        # Recent delinquency flag
        if 'months_since_last_delinquency' in df.columns:
            df['recent_delinquency'] = (
                (df['months_since_last_delinquency'] < 12) &
                (df['months_since_last_delinquency'].notna())
            ).astype(int)

        # Has public records
        if 'public_records' in df.columns:
            df['has_public_records'] = (df['public_records'] > 0).astype(int)

        # Has bankruptcies
        if 'public_record_bankruptcies' in df.columns:
            df['has_bankruptcy'] = (df['public_record_bankruptcies'] > 0).astype(int)

        return df

    def create_utilization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create utilization-related features

        Args:
            df: Input dataframe

        Returns:
            DataFrame with additional utilization features
        """
        df = df.copy()

        # Revolving balance to income
        if 'revolving_balance' in df.columns and 'annual_income' in df.columns:
            df['revolving_to_income'] = (
                df['revolving_balance'] / df['annual_income'].replace(0, np.nan)
            )

        # Total debt to income (including mortgage)
        if 'total_current_balance' in df.columns and 'annual_income' in df.columns:
            df['total_debt_to_income'] = (
                df['total_current_balance'] / df['annual_income'].replace(0, np.nan)
            )

        # High utilization flag
        if 'revolving_utilization' in df.columns:
            df['high_revolving_util'] = (df['revolving_utilization'] > 75).astype(int)

        return df

    def create_inquiry_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create inquiry-related features

        Args:
            df: Input dataframe

        Returns:
            DataFrame with additional inquiry features
        """
        df = df.copy()

        # Multiple recent inquiries
        if 'inquiries_last_6months' in df.columns:
            df['multiple_inquiries_6m'] = (df['inquiries_last_6months'] > 2).astype(int)

        # Recent inquiry flag
        if 'months_since_recent_inquiry' in df.columns:
            df['very_recent_inquiry'] = (
                (df['months_since_recent_inquiry'] < 3) &
                (df['months_since_recent_inquiry'].notna())
            ).astype(int)

        return df

    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from categorical variables

        Args:
            df: Input dataframe

        Returns:
            DataFrame with additional categorical features
        """
        df = df.copy()

        # Grade to numeric
        if 'grade' in df.columns:
            grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
            df['grade_numeric'] = df['grade'].map(grade_map)

        # Term to numeric
        if 'term' in df.columns:
            df['term_months'] = df['term'].str.extract(r'(\d+)').astype(float)

        # Employment length to numeric
        if 'employment_length' in df.columns:
            emp_map = {
                '< 1 year': 0,
                '1 year': 1,
                '2 years': 2,
                '3 years': 3,
                '4 years': 4,
                '5 years': 5,
                '6 years': 6,
                '7 years': 7,
                '8 years': 8,
                '9 years': 9,
                '10+ years': 10
            }
            df['employment_years'] = df['employment_length'].map(emp_map)

        # Home ownership binary
        if 'home_ownership' in df.columns:
            df['is_homeowner'] = df['home_ownership'].isin(['OWN', 'MORTGAGE']).astype(int)

        # Income verification binary
        if 'verification_status' in df.columns:
            df['is_income_verified'] = (
                df['verification_status'].isin(['Verified', 'Source Verified'])
            ).astype(int)

        # High risk loan purposes
        if 'loan_purpose' in df.columns:
            df['is_debt_consolidation'] = (
                df['loan_purpose'].isin(['debt_consolidation', 'credit_card'])
            ).astype(int)

        return df

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features

        Args:
            df: Input dataframe

        Returns:
            DataFrame with all engineered features
        """
        df = self.create_loan_features(df)
        df = self.create_credit_features(df)
        df = self.create_delinquency_features(df)
        df = self.create_utilization_features(df)
        df = self.create_inquiry_features(df)
        df = self.create_categorical_features(df)

        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: dict = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset

        Args:
            df: Input dataframe
            strategy: Dictionary mapping column names to imputation strategies

        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()

        if strategy is None:
            # Default strategy: median for numeric, mode for categorical
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0)
        else:
            for col, method in strategy.items():
                if col in df.columns:
                    if method == 'median':
                        df[col] = df[col].fillna(df[col].median())
                    elif method == 'mean':
                        df[col] = df[col].fillna(df[col].mean())
                    elif method == 'mode':
                        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0)
                    elif method == 'zero':
                        df[col] = df[col].fillna(0)
                    elif isinstance(method, (int, float, str)):
                        df[col] = df[col].fillna(method)

        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers from specified columns

        Args:
            df: Input dataframe
            columns: List of columns to check for outliers
            method: Method to use ('iqr' or 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]

        return df
