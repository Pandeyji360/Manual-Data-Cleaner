import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta
import string
import uuid
import argparse

def generate_random_date(start_date, end_date):
    """Generate a random date between start_date and end_date."""
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    return start_date + timedelta(days=random_number_of_days)

def generate_random_email(first_name, last_name, domains=None):
    """Generate a random email using first and last name."""
    if domains is None:
        domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'example.com']
    
    domain = random.choice(domains)
    separators = ['', '.', '_']
    separator = random.choice(separators)
    
    # Sometimes introduce errors in email format
    has_error = random.random() < 0.05  # 5% chance of error
    
    if has_error:
        error_types = ['missing_at', 'invalid_domain', 'double_dot', 'no_domain']
        error_type = random.choice(error_types)
        
        if error_type == 'missing_at':
            return f"{first_name.lower()}{separator}{last_name.lower()}{domain}"
        elif error_type == 'invalid_domain':
            return f"{first_name.lower()}{separator}{last_name.lower()}@{domain.replace('.com', '')}"
        elif error_type == 'double_dot':
            return f"{first_name.lower()}{separator}{last_name.lower()}@{domain.replace('.com', '..com')}"
        elif error_type == 'no_domain':
            return f"{first_name.lower()}{separator}{last_name.lower()}@"
    
    return f"{first_name.lower()}{separator}{last_name.lower()}@{domain}"

def generate_random_phone(country_code=None):
    """Generate a random phone number."""
    if country_code is None:
        country_code = random.choice(['', '+1', '+44', '+91', '+61', '+86'])
    
    # Generate a 10-digit phone number
    phone_number = ''.join(random.choices(string.digits, k=10))
    
    # Sometimes introduce errors in phone format
    has_error = random.random() < 0.05  # 5% chance of error
    
    if has_error:
        error_types = ['too_short', 'has_letters', 'invalid_format']
        error_type = random.choice(error_types)
        
        if error_type == 'too_short':
            phone_number = phone_number[:7]
        elif error_type == 'has_letters':
            # Replace some digits with letters
            char_positions = random.sample(range(len(phone_number)), random.randint(1, 3))
            phone_list = list(phone_number)
            for pos in char_positions:
                phone_list[pos] = random.choice(string.ascii_letters)
            phone_number = ''.join(phone_list)
        elif error_type == 'invalid_format':
            # Remove some digits and add invalid separators
            phone_list = list(phone_number)
            phone_list.insert(random.randint(0, len(phone_list)), random.choice('?!#'))
            phone_number = ''.join(phone_list)
    
    return f"{country_code}{phone_number}"

def generate_user_data(num_rows=100000, error_rate=0.1, include_missing=True):
    """
    Generate a DataFrame with user data.
    
    Args:
        num_rows (int): Number of rows to generate
        error_rate (float): Probability of introducing errors in the data (0-1)
        include_missing (bool): Whether to include missing values
    
    Returns:
        pd.DataFrame: DataFrame with generated user data
    """
    # Define first names
    first_names = [
        'James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda', 'William', 'Elizabeth',
        'David', 'Barbara', 'Richard', 'Susan', 'Joseph', 'Jessica', 'Thomas', 'Sarah', 'Charles', 'Karen',
        'Christopher', 'Nancy', 'Daniel', 'Lisa', 'Matthew', 'Margaret', 'Anthony', 'Betty', 'Mark', 'Sandra',
        'Donald', 'Ashley', 'Steven', 'Dorothy', 'Paul', 'Kimberly', 'Andrew', 'Emily', 'Joshua', 'Donna',
        'Kenneth', 'Michelle', 'Kevin', 'Carol', 'Brian', 'Amanda', 'George', 'Melissa', 'Edward', 'Deborah',
        'Ronald', 'Stephanie', 'Timothy', 'Rebecca', 'Jason', 'Sharon', 'Jeffrey', 'Laura', 'Ryan', 'Cynthia',
        'Jacob', 'Kathleen', 'Gary', 'Amy', 'Nicholas', 'Shirley', 'Eric', 'Anna', 'Jonathan', 'Angela'
    ]
    
    # Define last names
    last_names = [
        'Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor',
        'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia', 'Martinez', 'Robinson',
        'Clark', 'Rodriguez', 'Lewis', 'Lee', 'Walker', 'Hall', 'Allen', 'Young', 'Hernandez', 'King',
        'Wright', 'Lopez', 'Hill', 'Scott', 'Green', 'Adams', 'Baker', 'Gonzalez', 'Nelson', 'Carter',
        'Mitchell', 'Perez', 'Roberts', 'Turner', 'Phillips', 'Campbell', 'Parker', 'Evans', 'Edwards', 'Collins',
        'Stewart', 'Sanchez', 'Morris', 'Rogers', 'Reed', 'Cook', 'Morgan', 'Bell', 'Murphy', 'Bailey',
        'Rivera', 'Cooper', 'Richardson', 'Cox', 'Howard', 'Ward', 'Torres', 'Peterson', 'Gray', 'Ramirez'
    ]
    
    # Define cities
    cities = [
        'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego',
        'Dallas', 'San Jose', 'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'San Francisco', 'Charlotte',
        'Indianapolis', 'Seattle', 'Denver', 'Washington', 'Boston', 'El Paso', 'Nashville', 'Detroit', 'Portland',
        'Memphis', 'Oklahoma City', 'Las Vegas', 'Louisville', 'Baltimore', 'Milwaukee', 'Albuquerque', 'Tucson',
        'Fresno', 'Sacramento', 'Mesa', 'Kansas City', 'Atlanta', 'Long Beach', 'Colorado Springs', 'Raleigh',
        'Miami', 'Virginia Beach', 'Omaha', 'Oakland', 'Minneapolis', 'Tulsa', 'Arlington', 'New Orleans',
        'Wichita', 'Cleveland', 'Tampa', 'Bakersfield', 'Aurora', 'Anaheim', 'Honolulu', 'Santa Ana', 'Riverside'
    ]
    
    # Define states
    states = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
    ]
    
    # Define subscription types
    subscription_types = ['Free', 'Basic', 'Premium', 'Enterprise', 'Custom']
    
    # Define status options
    status_options = ['Active', 'Inactive', 'Pending', 'Suspended', 'Canceled']
    
    # Define data source options
    data_sources = ['Website', 'Mobile App', 'Referral', 'Social Media', 'Email Campaign', 'Partnership', 'Other']
    
    # Generate data
    data = {
        'user_id': [str(uuid.uuid4()) for _ in range(num_rows)],
        'first_name': [random.choice(first_names) for _ in range(num_rows)],
        'last_name': [random.choice(last_names) for _ in range(num_rows)],
        'age': [random.randint(18, 80) for _ in range(num_rows)],
        'join_date': [generate_random_date(datetime(2015, 1, 1), datetime.now()).strftime('%Y-%m-%d') for _ in range(num_rows)],
        'email': [],  # Will fill later with generated emails
        'phone': [generate_random_phone() for _ in range(num_rows)],
        'city': [random.choice(cities) for _ in range(num_rows)],
        'state': [random.choice(states) for _ in range(num_rows)],
        'subscription_type': [random.choice(subscription_types) for _ in range(num_rows)],
        'monthly_payment': [round(random.uniform(0, 200), 2) for _ in range(num_rows)],
        'total_spend': [],  # Will calculate based on join date and monthly payment
        'last_login': [generate_random_date(datetime(2020, 1, 1), datetime.now()).strftime('%Y-%m-%d') for _ in range(num_rows)],
        'login_count': [random.randint(1, 1000) for _ in range(num_rows)],
        'status': [random.choice(status_options) for _ in range(num_rows)],
        'referred_users': [random.randint(0, 20) for _ in range(num_rows)],
        'data_source': [random.choice(data_sources) for _ in range(num_rows)],
        'satisfaction_score': [random.randint(1, 10) for _ in range(num_rows)]
    }
    
    # Generate emails based on first and last names
    for i in range(num_rows):
        data['email'].append(generate_random_email(data['first_name'][i], data['last_name'][i]))
    
    # Calculate total spend based on join date and monthly payment
    for i in range(num_rows):
        join_date = datetime.strptime(data['join_date'][i], '%Y-%m-%d')
        current_date = datetime.now()
        months = (current_date.year - join_date.year) * 12 + (current_date.month - join_date.month)
        
        # Add some randomness to months (some users might have paused subscription)
        effective_months = max(1, months - random.randint(0, int(months * 0.2)))
        
        # Calculate total spend
        total_spend = data['monthly_payment'][i] * effective_months
        
        # Add some noise to total spend
        noise_factor = random.uniform(0.9, 1.1)
        data['total_spend'].append(round(total_spend * noise_factor, 2))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Introduce missing values if requested
    if include_missing:
        # Columns that can have missing values (excluding user_id)
        missable_columns = [col for col in df.columns if col != 'user_id']
        
        # Introduce missing values randomly
        for col in missable_columns:
            # Determine how many values to set as missing (between 1% and 5%)
            missing_count = int(num_rows * random.uniform(0.01, 0.05))
            
            # Randomly select indices to set as missing
            missing_indices = random.sample(range(num_rows), missing_count)
            
            # Set values as missing
            df.loc[missing_indices, col] = np.nan
    
    # Introduce errors in the data
    if error_rate > 0:
        # Number of errors to introduce
        num_errors = int(num_rows * error_rate)
        
        for _ in range(num_errors):
            # Select a random row
            row_idx = random.randint(0, num_rows - 1)
            
            # Select a random column (excluding user_id)
            col_name = random.choice([col for col in df.columns if col != 'user_id'])
            
            # Introduce an error based on the column type
            if col_name in ['age', 'login_count', 'referred_users', 'satisfaction_score']:
                # For integer columns, introduce negative values or extremely high values
                error_type = random.choice(['negative', 'high_value'])
                
                if error_type == 'negative':
                    df.at[row_idx, col_name] = -random.randint(1, 100)
                else:  # high_value
                    df.at[row_idx, col_name] = random.randint(1000, 100000)
            
            elif col_name in ['monthly_payment', 'total_spend']:
                # For float columns, introduce negative values or extremely high values
                error_type = random.choice(['negative', 'high_value'])
                
                if error_type == 'negative':
                    df.at[row_idx, col_name] = -random.uniform(1, 1000)
                else:  # high_value
                    df.at[row_idx, col_name] = random.uniform(10000, 1000000)
            
            elif col_name in ['join_date', 'last_login']:
                # For date columns, introduce invalid dates or future dates
                error_type = random.choice(['invalid_format', 'future_date', 'ancient_date'])
                
                if error_type == 'invalid_format':
                    df.at[row_idx, col_name] = random.choice(['2022/13/45', '01-01-2022', 'not a date', '2022-1-1'])
                elif error_type == 'future_date':
                    future_date = datetime.now() + timedelta(days=random.randint(1, 1000))
                    df.at[row_idx, col_name] = future_date.strftime('%Y-%m-%d')
                else:
                    ancient_date = datetime(1900, 1, 1) + timedelta(days=random.randint(0, 365 * 50))
                    df.at[row_idx, col_name] = ancient_date.strftime('%Y-%m-%d')
            
            elif col_name in ['city', 'first_name', 'last_name']:
                # For name columns, introduce numbers or special characters
                # First ensure we're working with string values
                current_value = str(df.at[row_idx, col_name])
                
                error_type = random.choice(['numbers', 'special_chars', 'very_long'])
                
                if error_type == 'numbers':
                    df.at[row_idx, col_name] = current_value + str(random.randint(100, 999))
                elif error_type == 'special_chars':
                    df.at[row_idx, col_name] = current_value + random.choice(['!', '@', '#', '$', '%'])
                else:
                    df.at[row_idx, col_name] = current_value * random.randint(2, 5)
            
            elif col_name == 'state':
                # For state column, introduce invalid state codes
                df.at[row_idx, col_name] = random.choice(['XX', 'YY', 'ZZ', '12', 'ABC'])
            
            elif col_name == 'subscription_type':
                # For subscription type, introduce invalid types
                df.at[row_idx, col_name] = random.choice(['Unknown', 'Trial-Error', 'FREE!!!', 'Type1'])
            
            elif col_name == 'status':
                # For status, introduce invalid statuses
                df.at[row_idx, col_name] = random.choice(['Unknown', 'Maybe', 'Both', '1', '*active*'])
    
    # Add duplicate rows (about 2% of data)
    num_duplicates = int(num_rows * 0.02)
    duplicate_indices = random.sample(range(num_rows), num_duplicates)
    
    for idx in duplicate_indices:
        # Add the same row again
        df = pd.concat([df, df.iloc[[idx]]], ignore_index=True)
    
    return df

def generate_transaction_data(num_rows=100000, error_rate=0.1, include_missing=True):
    """
    Generate a DataFrame with transaction data.
    
    Args:
        num_rows (int): Number of rows to generate
        error_rate (float): Probability of introducing errors in the data (0-1)
        include_missing (bool): Whether to include missing values
    
    Returns:
        pd.DataFrame: DataFrame with generated transaction data
    """
    # Generate transaction IDs
    transaction_ids = [str(uuid.uuid4()) for _ in range(num_rows)]
    
    # Generate user IDs (some will be repeated to simulate multiple transactions per user)
    num_users = int(num_rows * 0.2)  # Assume about 20% of the number of transactions
    user_ids = [str(uuid.uuid4()) for _ in range(num_users)]
    
    # Transaction types
    transaction_types = ['Purchase', 'Refund', 'Subscription', 'Upgrade', 'Downgrade', 'Credit', 'Gift']
    
    # Payment methods
    payment_methods = ['Credit Card', 'PayPal', 'Bank Transfer', 'Apple Pay', 'Google Pay', 'Cryptocurrency', 'Gift Card']
    
    # Product categories
    product_categories = ['Electronics', 'Clothing', 'Books', 'Home & Kitchen', 'Sports', 'Beauty', 'Toys', 'Software', 'Music', 'Food']
    
    # Status options
    status_options = ['Completed', 'Pending', 'Failed', 'Refunded', 'Disputed']
    
    # Countries
    countries = ['USA', 'Canada', 'UK', 'Australia', 'Germany', 'France', 'Japan', 'China', 'Brazil', 'India', 'Mexico']
    
    # Currency codes
    currency_codes = ['USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY', 'CNY', 'BRL', 'INR', 'MXN']
    
    # Generate data
    data = {
        'transaction_id': transaction_ids,
        'user_id': [random.choice(user_ids) for _ in range(num_rows)],
        'transaction_date': [generate_random_date(datetime(2019, 1, 1), datetime.now()).strftime('%Y-%m-%d') for _ in range(num_rows)],
        'transaction_time': [f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}" for _ in range(num_rows)],
        'transaction_type': [random.choice(transaction_types) for _ in range(num_rows)],
        'amount': [round(random.uniform(5, 500), 2) for _ in range(num_rows)],
        'currency': [random.choice(currency_codes) for _ in range(num_rows)],
        'payment_method': [random.choice(payment_methods) for _ in range(num_rows)],
        'product_id': [str(uuid.uuid4()) for _ in range(num_rows)],
        'product_category': [random.choice(product_categories) for _ in range(num_rows)],
        'quantity': [random.randint(1, 10) for _ in range(num_rows)],
        'unit_price': [],  # Will calculate based on amount and quantity
        'status': [random.choice(status_options) for _ in range(num_rows)],
        'country': [random.choice(countries) for _ in range(num_rows)],
        'shipping_cost': [round(random.uniform(0, 50), 2) for _ in range(num_rows)],
        'tax_amount': [round(random.uniform(0, 30), 2) for _ in range(num_rows)],
        'discount_amount': [round(random.uniform(0, 20), 2) for _ in range(num_rows)],
        'is_first_purchase': [random.choice([True, False]) for _ in range(num_rows)],
        'processing_time_ms': [random.randint(100, 5000) for _ in range(num_rows)]
    }
    
    # Calculate unit price based on amount and quantity
    for i in range(num_rows):
        # Add some noise to make it more realistic (not always exactly amount/quantity)
        # For refunds, the amount is negative
        amount = data['amount'][i]
        if data['transaction_type'][i] == 'Refund':
            amount = -amount
            data['amount'][i] = amount
        
        unit_price = amount / data['quantity'][i]
        
        # Add a small random variation
        noise_factor = random.uniform(0.95, 1.05)
        data['unit_price'].append(round(unit_price * noise_factor, 2))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by transaction date and time
    df['datetime'] = pd.to_datetime(df['transaction_date'] + ' ' + df['transaction_time'])
    df = df.sort_values('datetime')
    df = df.drop('datetime', axis=1)
    
    # Introduce missing values if requested
    if include_missing:
        # Columns that can have missing values (excluding transaction_id)
        missable_columns = [col for col in df.columns if col != 'transaction_id']
        
        # Introduce missing values randomly
        for col in missable_columns:
            # Determine how many values to set as missing (between 1% and 5%)
            missing_count = int(num_rows * random.uniform(0.01, 0.05))
            
            # Randomly select indices to set as missing
            missing_indices = random.sample(range(num_rows), missing_count)
            
            # Set values as missing
            df.loc[missing_indices, col] = np.nan
    
    # Introduce errors in the data
    if error_rate > 0:
        # Number of errors to introduce
        num_errors = int(num_rows * error_rate)
        
        for _ in range(num_errors):
            # Select a random row
            row_idx = random.randint(0, num_rows - 1)
            
            # Select a random column (excluding transaction_id)
            col_name = random.choice([col for col in df.columns if col != 'transaction_id'])
            
            # Introduce an error based on the column type
            if col_name in ['quantity', 'processing_time_ms']:
                # For integer columns, introduce negative values or extremely high values
                error_type = random.choice(['negative', 'high_value'])
                
                if error_type == 'negative':
                    df.at[row_idx, col_name] = -random.randint(1, 100)
                else:  # high_value
                    df.at[row_idx, col_name] = random.randint(1000, 100000)
            
            elif col_name in ['amount', 'unit_price', 'shipping_cost', 'tax_amount', 'discount_amount']:
                # For float columns, introduce negative values (where inappropriate) or extremely high values
                error_type = random.choice(['negative', 'high_value'])
                
                if error_type == 'negative' and col_name != 'amount':  # amount can be negative for refunds
                    df.at[row_idx, col_name] = -random.uniform(1, 100)
                else:  # high_value
                    df.at[row_idx, col_name] = random.uniform(1000, 10000)
            
            elif col_name in ['transaction_date']:
                # For date columns, introduce invalid dates or future dates
                error_type = random.choice(['invalid_format', 'future_date', 'ancient_date'])
                
                if error_type == 'invalid_format':
                    df.at[row_idx, col_name] = random.choice(['2022/13/45', '01-01-2022', 'not a date', '2022-1-1'])
                elif error_type == 'future_date':
                    future_date = datetime.now() + timedelta(days=random.randint(1, 1000))
                    df.at[row_idx, col_name] = future_date.strftime('%Y-%m-%d')
                else:
                    ancient_date = datetime(1900, 1, 1) + timedelta(days=random.randint(0, 365 * 50))
                    df.at[row_idx, col_name] = ancient_date.strftime('%Y-%m-%d')
            
            elif col_name == 'transaction_time':
                # For time columns, introduce invalid times
                error_type = random.choice(['invalid_format', 'out_of_range'])
                
                if error_type == 'invalid_format':
                    df.at[row_idx, col_name] = random.choice(['25:00:00', '12:65:00', '12:30', 'noon'])
                else:
                    df.at[row_idx, col_name] = f"{random.randint(24, 99)}:{random.randint(60, 99)}:{random.randint(60, 99)}"
            
            elif col_name == 'currency':
                # For currency, introduce invalid codes
                df.at[row_idx, col_name] = random.choice(['US', 'DOLLAR', '123', 'Bitcoin', 'ERROR'])
            
            elif col_name == 'status':
                # For status, introduce invalid statuses
                df.at[row_idx, col_name] = random.choice(['Unknown', 'Maybe', 'Processing...', '1', 'done!'])
    
    # Add duplicate rows (about 2% of data)
    num_duplicates = int(num_rows * 0.02)
    duplicate_indices = random.sample(range(num_rows), num_duplicates)
    
    for idx in duplicate_indices:
        # Add the same row again
        df = pd.concat([df, df.iloc[[idx]]], ignore_index=True)
    
    return df

def generate_product_data(num_rows=10000, error_rate=0.1, include_missing=True):
    """
    Generate a DataFrame with product data.
    
    Args:
        num_rows (int): Number of rows to generate
        error_rate (float): Probability of introducing errors in the data (0-1)
        include_missing (bool): Whether to include missing values
    
    Returns:
        pd.DataFrame: DataFrame with generated product data
    """
    # Product names - base words
    adjectives = [
        'Premium', 'Deluxe', 'Advanced', 'Smart', 'Ultra', 'Super', 'Eco', 'Professional',
        'Classic', 'Elite', 'Luxury', 'Budget', 'Compact', 'Portable', 'Wireless', 'Digital',
        'Modern', 'Vintage', 'Organic', 'Handmade', 'Custom', 'Innovative', 'Traditional', 'High-End'
    ]
    
    product_types = [
        'Laptop', 'Smartphone', 'Headphones', 'Speaker', 'Camera', 'TV', 'Monitor', 'Tablet',
        'Keyboard', 'Mouse', 'Printer', 'Charger', 'Cable', 'Case', 'Stand', 'Adapter',
        'Watch', 'Fitness Tracker', 'Game Console', 'Router', 'Hard Drive', 'SSD', 'Memory Card', 'USB Drive',
        'T-Shirt', 'Jeans', 'Dress', 'Shoes', 'Hat', 'Jacket', 'Sweater', 'Socks',
        'Book', 'Notebook', 'Pen', 'Pencil', 'Marker', 'Calendar', 'Planner', 'Sticker',
        'Chair', 'Table', 'Desk', 'Lamp', 'Rug', 'Mirror', 'Clock', 'Pillow',
        'Shampoo', 'Conditioner', 'Soap', 'Lotion', 'Perfume', 'Lipstick', 'Mascara', 'Nail Polish'
    ]
    
    # Brands
    brands = [
        'TechPro', 'GlobalElite', 'NextGen', 'FutureTech', 'InnovateX', 'PrimeTech', 'SmartLife', 'EcoSolutions',
        'LuxuryGoods', 'UrbanStyle', 'NaturalChoice', 'HomeComfort', 'TravelEssentials', 'FitnessFirst', 'GourmetDelight',
        'CreativeMinds', 'DigitalDreams', 'ClassicDesign', 'ModernLiving', 'PurePlanet', 'ElitePerformance', 'SimpleSolutions'
    ]
    
    # Categories
    categories = [
        'Electronics', 'Computers', 'Mobile Phones', 'Audio', 'Photography', 'TV & Home Cinema', 'Computer Accessories',
        'Clothing', 'Footwear', 'Accessories', 'Jewelry', 'Watches',
        'Books', 'Stationery', 'Office Supplies',
        'Furniture', 'Home Decor', 'Kitchen', 'Bathroom', 'Bedroom',
        'Beauty', 'Personal Care', 'Health & Wellness',
        'Sports', 'Outdoors', 'Fitness',
        'Toys', 'Games', 'Hobbies',
        'Food', 'Drinks', 'Snacks'
    ]
    
    # Subcategories (simplified)
    subcategories = {
        'Electronics': ['Laptops', 'Desktops', 'Tablets', 'Smartphones', 'Wearables', 'Cameras'],
        'Computers': ['Gaming PC', 'Business Laptop', 'Workstation', 'Chromebook', 'All-in-One'],
        'Mobile Phones': ['Android', 'iOS', 'Feature Phones', 'Accessories', 'Cases'],
        'Audio': ['Headphones', 'Earbuds', 'Speakers', 'Soundbars', 'Microphones'],
        'Clothing': ['Shirts', 'Pants', 'Dresses', 'Outerwear', 'Activewear', 'Underwear'],
        'Footwear': ['Sneakers', 'Boots', 'Sandals', 'Formal Shoes', 'Athletic Shoes'],
        'Books': ['Fiction', 'Non-Fiction', 'Reference', 'Textbooks', 'Comics', 'Magazines'],
        'Furniture': ['Chairs', 'Tables', 'Sofas', 'Beds', 'Storage', 'Outdoor'],
        'Beauty': ['Skincare', 'Makeup', 'Fragrance', 'Hair Care', 'Nail Care']
    }
    
    # Suppliers
    suppliers = [
        'Global Imports Inc.', 'Quality Products Ltd.', 'Direct Manufacturer', 'Wholesale Solutions', 
        'International Trade Co.', 'Factory Direct', 'Premium Suppliers', 'Eco Sourcing', 
        'National Distribution', 'Regional Imports'
    ]
    
    # Status options
    status_options = ['In Stock', 'Out of Stock', 'Discontinued', 'Coming Soon', 'Backordered']
    
    # Generate product IDs
    product_ids = [str(uuid.uuid4()) for _ in range(num_rows)]
    
    # Generate SKUs
    skus = [f"SKU-{random.randint(10000, 99999)}-{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}" for _ in range(num_rows)]
    
    # Generate product names
    product_names = []
    for _ in range(num_rows):
        adjective = random.choice(adjectives)
        product_type = random.choice(product_types)
        product_names.append(f"{adjective} {product_type}")
    
    # Generate category and subcategory pairs
    all_categories = []
    all_subcategories = []
    
    for _ in range(num_rows):
        category = random.choice(categories)
        all_categories.append(category)
        
        if category in subcategories:
            subcategory = random.choice(subcategories[category])
        else:
            subcategory = 'General'  # Default subcategory if not found
        
        all_subcategories.append(subcategory)
    
    # Generate data
    data = {
        'product_id': product_ids,
        'sku': skus,
        'product_name': product_names,
        'description': [f"This is a {name.lower()} with great features." for name in product_names],
        'brand': [random.choice(brands) for _ in range(num_rows)],
        'category': all_categories,
        'subcategory': all_subcategories,
        'price': [round(random.uniform(10, 1000), 2) for _ in range(num_rows)],
        'cost': [],  # Will calculate as percentage of price
        'weight_kg': [round(random.uniform(0.1, 20), 2) for _ in range(num_rows)],
        'dimensions_cm': [f"{random.randint(5, 100)}x{random.randint(5, 100)}x{random.randint(2, 50)}" for _ in range(num_rows)],
        'stock_quantity': [random.randint(0, 1000) for _ in range(num_rows)],
        'reorder_level': [random.randint(5, 100) for _ in range(num_rows)],
        'supplier': [random.choice(suppliers) for _ in range(num_rows)],
        'rating': [round(random.uniform(1, 5), 1) for _ in range(num_rows)],
        'review_count': [random.randint(0, 1000) for _ in range(num_rows)],
        'status': [],  # Will determine based on stock quantity
        'is_featured': [random.choice([True, False]) for _ in range(num_rows)],
        'created_date': [generate_random_date(datetime(2018, 1, 1), datetime.now()).strftime('%Y-%m-%d') for _ in range(num_rows)],
        'modified_date': []  # Will calculate as a later date than created_date
    }
    
    # Calculate cost (usually 40-70% of price)
    for i in range(num_rows):
        cost_factor = random.uniform(0.4, 0.7)
        data['cost'].append(round(data['price'][i] * cost_factor, 2))
    
    # Determine status based on stock quantity
    for i in range(num_rows):
        stock = data['stock_quantity'][i]
        reorder = data['reorder_level'][i]
        
        if stock == 0:
            data['status'].append('Out of Stock')
        elif stock < reorder:
            data['status'].append(random.choice(['Low Stock', 'Backordered']))
        else:
            data['status'].append('In Stock')
        
        # Override some statuses randomly
        if random.random() < 0.05:  # 5% chance
            data['status'][i] = random.choice(['Discontinued', 'Coming Soon'])
    
    # Calculate modified date (between created_date and now)
    for i in range(num_rows):
        created_date = datetime.strptime(data['created_date'][i], '%Y-%m-%d')
        modified_date = generate_random_date(created_date, datetime.now())
        data['modified_date'].append(modified_date.strftime('%Y-%m-%d'))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Introduce missing values if requested
    if include_missing:
        # Columns that can have missing values (excluding product_id and sku)
        missable_columns = [col for col in df.columns if col not in ['product_id', 'sku']]
        
        # Introduce missing values randomly
        for col in missable_columns:
            # Determine how many values to set as missing (between 1% and 5%)
            missing_count = int(num_rows * random.uniform(0.01, 0.05))
            
            # Randomly select indices to set as missing
            missing_indices = random.sample(range(num_rows), missing_count)
            
            # Set values as missing
            df.loc[missing_indices, col] = np.nan
    
    # Introduce errors in the data
    if error_rate > 0:
        # Number of errors to introduce
        num_errors = int(num_rows * error_rate)
        
        for _ in range(num_errors):
            # Select a random row
            row_idx = random.randint(0, num_rows - 1)
            
            # Select a random column (excluding product_id)
            col_name = random.choice([col for col in df.columns if col != 'product_id'])
            
            # Introduce an error based on the column type
            if col_name in ['stock_quantity', 'reorder_level', 'review_count']:
                # For integer columns, introduce negative values or extremely high values
                error_type = random.choice(['negative', 'high_value'])
                
                if error_type == 'negative':
                    df.at[row_idx, col_name] = -random.randint(1, 100)
                else:  # high_value
                    df.at[row_idx, col_name] = random.randint(10000, 1000000)
            
            elif col_name in ['price', 'cost', 'weight_kg', 'rating']:
                # For float columns, introduce negative values or extremely high values
                error_type = random.choice(['negative', 'high_value'])
                
                if error_type == 'negative':
                    df.at[row_idx, col_name] = -random.uniform(1, 100)
                else:  # high_value
                    df.at[row_idx, col_name] = random.uniform(10000, 1000000)
            
            elif col_name in ['created_date', 'modified_date']:
                # For date columns, introduce invalid dates or future dates
                error_type = random.choice(['invalid_format', 'future_date', 'ancient_date'])
                
                if error_type == 'invalid_format':
                    df.at[row_idx, col_name] = random.choice(['2022/13/45', '01-01-2022', 'not a date', '2022-1-1'])
                elif error_type == 'future_date':
                    future_date = datetime.now() + timedelta(days=random.randint(1, 1000))
                    df.at[row_idx, col_name] = future_date.strftime('%Y-%m-%d')
                else:
                    ancient_date = datetime(1900, 1, 1) + timedelta(days=random.randint(0, 365 * 50))
                    df.at[row_idx, col_name] = ancient_date.strftime('%Y-%m-%d')
            
            elif col_name == 'dimensions_cm':
                # For dimensions, introduce invalid formats
                error_type = random.choice(['invalid_format', 'negative', 'words'])
                
                if error_type == 'invalid_format':
                    df.at[row_idx, col_name] = random.choice(['10-20-30', '10/20/30', '10 cm x 20 cm x 30 cm'])
                elif error_type == 'negative':
                    df.at[row_idx, col_name] = f"{-random.randint(1, 50)}x{-random.randint(1, 50)}x{-random.randint(1, 50)}"
                else:
                    df.at[row_idx, col_name] = 'width x height x depth'
            
            elif col_name == 'status':
                # For status, introduce invalid statuses
                df.at[row_idx, col_name] = random.choice(['Unknown', 'Maybe', 'Available?', '1', 'CHECK!'])
    
    # Add duplicate rows (about 2% of data)
    num_duplicates = int(num_rows * 0.02)
    duplicate_indices = random.sample(range(num_rows), num_duplicates)
    
    for idx in duplicate_indices:
        # Add the same row again
        df = pd.concat([df, df.iloc[[idx]]], ignore_index=True)
    
    return df

def generate_customer_support_data(num_rows=50000, error_rate=0.1, include_missing=True):
    """
    Generate a DataFrame with customer support data.
    
    Args:
        num_rows (int): Number of rows to generate
        error_rate (float): Probability of introducing errors in the data (0-1)
        include_missing (bool): Whether to include missing values
    
    Returns:
        pd.DataFrame: DataFrame with generated customer support data
    """
    # Generate ticket IDs
    ticket_ids = [f"TICKET-{random.randint(10000, 99999)}" for _ in range(num_rows)]
    
    # Generate user IDs (some will be repeated to simulate multiple tickets per user)
    num_users = int(num_rows * 0.3)  # Assume about 30% of the number of tickets
    user_ids = [str(uuid.uuid4()) for _ in range(num_users)]
    
    # Issue types
    issue_types = [
        'Account Access', 'Billing Problem', 'Technical Issue', 'Product Information', 'Return Request',
        'Shipping Delay', 'Product Defect', 'Feature Request', 'Website Error', 'Payment Failure',
        'Password Reset', 'Account Deletion', 'Subscription Cancel', 'Product Availability', 'General Inquiry'
    ]
    
    # Products
    products = [
        'Mobile App', 'Website', 'Premium Subscription', 'Basic Subscription', 'Product Model A',
        'Product Model B', 'Product Model C', 'Service Package X', 'Service Package Y', 'API Access',
        'Analytics Dashboard', 'Browser Extension', 'Desktop Application', 'Integration Services', 'Cloud Storage'
    ]
    
    # Priorities
    priorities = ['Low', 'Medium', 'High', 'Critical']
    
    # Statuses
    statuses = ['Open', 'In Progress', 'Waiting for Customer', 'Waiting for Third-party', 'Resolved', 'Closed']
    
    # Support channels
    channels = ['Email', 'Phone', 'Live Chat', 'Social Media', 'In-App', 'Website Form', 'API', 'In Person']
    
    # Customer satisfaction ratings (1-5)
    satisfaction_ratings = [1, 2, 3, 4, 5, None]  # Include None for unrated tickets
    
    # Agent names (first name only for simplicity)
    agent_names = [
        'John', 'Emma', 'Michael', 'Sophia', 'William', 'Olivia', 'James', 'Ava', 'Benjamin', 'Isabella',
        'Alexander', 'Mia', 'Daniel', 'Charlotte', 'Matthew', 'Amelia', 'Joseph', 'Harper', 'David', 'Evelyn'
    ]
    
    # Departments
    departments = ['Technical Support', 'Billing', 'Account Management', 'Product Support', 'Returns', 'General Support']
    
    # Generate data
    data = {
        'ticket_id': ticket_ids,
        'user_id': [random.choice(user_ids) for _ in range(num_rows)],
        'created_date': [generate_random_date(datetime(2020, 1, 1), datetime.now()).strftime('%Y-%m-%d') for _ in range(num_rows)],
        'created_time': [f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}" for _ in range(num_rows)],
        'issue_type': [random.choice(issue_types) for _ in range(num_rows)],
        'product': [random.choice(products) for _ in range(num_rows)],
        'subject': [],  # Will generate based on issue type
        'description': [],  # Will generate based on issue type and product
        'priority': [random.choice(priorities) for _ in range(num_rows)],
        'status': [],  # Will determine partly based on created date
        'channel': [random.choice(channels) for _ in range(num_rows)],
        'assigned_to': [random.choice(agent_names) for _ in range(num_rows)],
        'department': [random.choice(departments) for _ in range(num_rows)],
        'first_response_time_min': [],  # Will calculate
        'resolution_time_hrs': [],  # Will calculate
        'satisfaction_rating': [random.choice(satisfaction_ratings) for _ in range(num_rows)],
        'notes': [],  # Will generate
        'is_reopened': [random.choice([True, False, False, False]) for _ in range(num_rows)],  # 25% chance of reopened
        'updates_count': [random.randint(1, 10) for _ in range(num_rows)]
    }
    
    # Generate subjects based on issue type
    subjects = {
        'Account Access': ['Cannot login', 'Account locked', 'Login error', 'Password not working'],
        'Billing Problem': ['Wrong charge', 'Double billing', 'Cannot update payment', 'Missing invoice'],
        'Technical Issue': ['App crashing', 'Error message', 'Feature not working', 'Performance issue'],
        'Product Information': ['Product specs', 'Compatibility question', 'Feature inquiry', 'Usage instructions'],
        'Return Request': ['Defective product', 'Wrong item', 'Not as described', 'Changed mind'],
        'Shipping Delay': ['Order not received', 'Tracking shows no movement', 'Estimated delivery passed', 'Wrong address'],
        'Product Defect': ['Not working properly', 'Broken on arrival', 'Missing parts', 'Quality issue'],
        'Feature Request': ['New feature suggestion', 'Improvement idea', 'Missing functionality', 'Enhancement request'],
        'Website Error': ['Page not loading', '404 error', 'Cannot checkout', 'Form submission error'],
        'Payment Failure': ['Card declined', 'Payment not processing', 'PayPal error', 'Billing verification needed'],
        'Password Reset': ['Cannot reset password', 'Reset link not received', 'Reset not working', 'Need password help'],
        'Account Deletion': ['Request to delete account', 'Privacy concern', 'How to delete account', 'GDPR request'],
        'Subscription Cancel': ['Cancel subscription', 'Downgrade plan', 'Refund request', 'Too expensive'],
        'Product Availability': ['Out of stock inquiry', 'Waitlist request', 'Restock timing', 'Alternative product'],
        'General Inquiry': ['General question', 'Information request', 'Help needed', 'Assistance required']
    }
    
    # Generate descriptions
    descriptions = {
        'Account Access': [
            'I cannot log into my account despite multiple attempts.',
            'My account is locked after several login attempts.',
            'I am getting an error message when trying to login.',
            'My password is not working even though I know it is correct.'
        ],
        'Billing Problem': [
            'I was charged incorrectly for my last purchase.',
            'I see two identical charges on my statement.',
            'I cannot update my payment method in my account.',
            'I have not received an invoice for my last payment.'
        ],
        'Technical Issue': [
            'The application keeps crashing when I try to use it.',
            'I am getting an error message that says: "Unexpected error occurred".',
            'The search feature is not returning any results.',
            'The website is extremely slow when I try to browse products.'
        ]
    }
    
    # Fill in the remaining descriptions with generic content
    for issue in issue_types:
        if issue not in descriptions:
            descriptions[issue] = [
                f'I am having an issue with {issue.lower()}.',
                f'Please help me resolve a problem related to {issue.lower()}.',
                f'I need assistance with {issue.lower()}.',
                f'There seems to be a problem with {issue.lower()} that I need help with.'
            ]
    
    # Generate notes
    notes_templates = [
        'Customer contacted via {channel}. Issue being investigated.',
        'Followed up with customer for more information.',
        'Escalated to {department} for specialized assistance.',
        'Issue identified: {issue_detail}. Working on resolution.',
        'Provided temporary workaround to customer.',
        'Customer informed that we are working on the issue.',
        'Issue resolved by {resolution_action}.',
        'Waiting for customer response.',
        'Third-party vendor contacted regarding this issue.',
        'Customer confirmed issue is resolved.'
    ]
    
    # Generate subjects, descriptions, and notes
    for i in range(num_rows):
        issue = data['issue_type'][i]
        product = data['product'][i]
        
        # Generate subject
        if issue in subjects:
            data['subject'].append(f"{random.choice(subjects[issue])} - {product}")
        else:
            data['subject'].append(f"Issue with {product}")
        
        # Generate description
        if issue in descriptions:
            data['description'].append(f"{random.choice(descriptions[issue])} This is related to {product}.")
        else:
            data['description'].append(f"I am having an issue with {product} related to {issue.lower()}. Please help.")
        
        # Generate notes
        note_template = random.choice(notes_templates)
        note = note_template.format(
            channel=data['channel'][i],
            department=data['department'][i],
            issue_detail=f"Problem with {product} {issue.lower()}",
            resolution_action=random.choice([
                'software update', 'configuration change', 'account reset',
                'providing detailed instructions', 'replacing the product'
            ])
        )
        data['notes'].append(note)
    
    # Calculate first response time (minutes)
    for i in range(num_rows):
        priority = data['priority'][i]
        
        # Higher priority tickets typically get faster responses
        if priority == 'Critical':
            base_time = random.randint(5, 30)
        elif priority == 'High':
            base_time = random.randint(15, 60)
        elif priority == 'Medium':
            base_time = random.randint(30, 120)
        else:  # Low
            base_time = random.randint(60, 240)
        
        # Add some variability
        variability = random.uniform(0.7, 1.3)
        first_response_time = int(base_time * variability)
        data['first_response_time_min'].append(first_response_time)
    
    # Calculate resolution time (hours)
    for i in range(num_rows):
        created_date = datetime.strptime(data['created_date'][i], '%Y-%m-%d')
        days_ago = (datetime.now() - created_date).days
        issue_type = data['issue_type'][i]
        priority = data['priority'][i]
        
        # Some tickets will still be open
        if days_ago < 2 or random.random() < 0.1:  # 10% chance of still being open for older tickets
            data['status'].append(random.choice(['Open', 'In Progress', 'Waiting for Customer', 'Waiting for Third-party']))
            data['resolution_time_hrs'].append(None)  # Still open, no resolution time
            continue
        
        # Ticket is resolved/closed
        data['status'].append(random.choice(['Resolved', 'Closed']))
        
        # Calculate resolution time based on issue type and priority
        if priority == 'Critical':
            base_time = random.randint(1, 12)  # 1-12 hours
        elif priority == 'High':
            base_time = random.randint(4, 24)  # 4-24 hours
        elif priority == 'Medium':
            base_time = random.randint(12, 72)  # 12-72 hours
        else:  # Low
            base_time = random.randint(24, 120)  # 24-120 hours
        
        # Certain issue types take longer
        if issue_type in ['Technical Issue', 'Product Defect']:
            base_time *= random.uniform(1.2, 1.5)
        
        # Add some variability
        variability = random.uniform(0.8, 1.2)
        resolution_time = int(base_time * variability)
        data['resolution_time_hrs'].append(resolution_time)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure consistent statuses (tickets with resolution time should be Resolved or Closed)
    for i, row in df.iterrows():
        if pd.notna(row['resolution_time_hrs']) and row['status'] not in ['Resolved', 'Closed']:
            df.at[i, 'status'] = random.choice(['Resolved', 'Closed'])
    
    # Introduce missing values if requested
    if include_missing:
        # Columns that can have missing values (excluding ticket_id)
        missable_columns = [col for col in df.columns if col != 'ticket_id']
        
        # Introduce missing values randomly
        for col in missable_columns:
            # Determine how many values to set as missing (between 1% and 5%)
            missing_count = int(num_rows * random.uniform(0.01, 0.05))
            
            # Randomly select indices to set as missing
            missing_indices = random.sample(range(num_rows), missing_count)
            
            # Set values as missing
            df.loc[missing_indices, col] = np.nan
    
    # Introduce errors in the data
    if error_rate > 0:
        # Number of errors to introduce
        num_errors = int(num_rows * error_rate)
        
        for _ in range(num_errors):
            # Select a random row
            row_idx = random.randint(0, num_rows - 1)
            
            # Select a random column (excluding ticket_id)
            col_name = random.choice([col for col in df.columns if col != 'ticket_id'])
            
            # Introduce an error based on the column type
            if col_name in ['first_response_time_min', 'resolution_time_hrs', 'updates_count']:
                # For numeric columns, introduce negative values or extremely high values
                error_type = random.choice(['negative', 'high_value', 'string'])
                
                if error_type == 'negative':
                    df.at[row_idx, col_name] = -random.randint(1, 100)
                elif error_type == 'high_value':
                    df.at[row_idx, col_name] = random.randint(1000, 10000)
                else:
                    df.at[row_idx, col_name] = random.choice(['N/A', 'unknown', 'error'])
            
            elif col_name in ['created_date']:
                # For date columns, introduce invalid dates or future dates
                error_type = random.choice(['invalid_format', 'future_date', 'ancient_date'])
                
                if error_type == 'invalid_format':
                    df.at[row_idx, col_name] = random.choice(['2022/13/45', '01-01-2022', 'not a date', '2022-1-1'])
                elif error_type == 'future_date':
                    future_date = datetime.now() + timedelta(days=random.randint(1, 1000))
                    df.at[row_idx, col_name] = future_date.strftime('%Y-%m-%d')
                else:
                    ancient_date = datetime(1900, 1, 1) + timedelta(days=random.randint(0, 365 * 50))
                    df.at[row_idx, col_name] = ancient_date.strftime('%Y-%m-%d')
            
            elif col_name == 'created_time':
                # For time columns, introduce invalid times
                error_type = random.choice(['invalid_format', 'out_of_range'])
                
                if error_type == 'invalid_format':
                    df.at[row_idx, col_name] = random.choice(['25:00:00', '12:65:00', '12:30', 'noon'])
                else:
                    df.at[row_idx, col_name] = f"{random.randint(24, 99)}:{random.randint(60, 99)}:{random.randint(60, 99)}"
            
            elif col_name == 'priority':
                # For priority, introduce invalid values
                df.at[row_idx, col_name] = random.choice(['URGENT!!!', 'low', 'Very High', '1', 'P1'])
            
            elif col_name == 'status':
                # For status, introduce invalid statuses
                df.at[row_idx, col_name] = random.choice(['URGENT HELP NEEDED', 'Waiting...', 'Almost done', '1', 'unresolved'])
    
    # Add duplicate rows (about 2% of data)
    num_duplicates = int(num_rows * 0.02)
    duplicate_indices = random.sample(range(num_rows), num_duplicates)
    
    for idx in duplicate_indices:
        # Add the same row again
        df = pd.concat([df, df.iloc[[idx]]], ignore_index=True)
    
    return df

def generate_website_analytics_data(num_rows=200000, error_rate=0.1, include_missing=True):
    """
    Generate a DataFrame with website analytics data.
    
    Args:
        num_rows (int): Number of rows to generate
        error_rate (float): Probability of introducing errors in the data (0-1)
        include_missing (bool): Whether to include missing values
    
    Returns:
        pd.DataFrame: DataFrame with generated website analytics data
    """
    # Date range for analytics
    start_date = datetime(2021, 1, 1)
    end_date = datetime.now()
    
    # Generate dates that skew toward more recent dates
    dates = []
    for _ in range(num_rows):
        # Create a distribution that favors more recent dates
        days_from_start = (end_date - start_date).days
        random_days = int(random.betavariate(2, 1) * days_from_start)
        date = start_date + timedelta(days=random_days)
        dates.append(date.strftime('%Y-%m-%d'))
    
    # Page URLs
    base_urls = [
        '/', '/about', '/products', '/contact', '/blog', '/faq', '/login', '/signup',
        '/profile', '/settings', '/cart', '/checkout', '/payment', '/confirmation',
        '/categories', '/search', '/support', '/terms', '/privacy', '/returns'
    ]
    
    # Add blog post URLs
    blog_urls = [f'/blog/post-{i}' for i in range(1, 51)]
    
    # Add product URLs
    product_urls = [f'/products/product-{i}' for i in range(1, 101)]
    
    # Combine URLs
    page_urls = base_urls + blog_urls + product_urls
    
    # Referrer domains
    referrer_domains = [
        'google.com', 'bing.com', 'facebook.com', 'twitter.com', 'instagram.com',
        'linkedin.com', 'youtube.com', 'reddit.com', 'yahoo.com', 'pinterest.com',
        't.co', 'news.ycombinator.com', 'producthunt.com', 'medium.com', 'quora.com',
        'duckduckgo.com', 'baidu.com', 'yandex.com', 'direct', 'email'
    ]
    
    # Generate full referrer URLs
    referrer_urls = []
    for domain in referrer_domains:
        if domain == 'direct':
            referrer_urls.append('direct')
        elif domain == 'email':
            referrer_urls.append('email')
        elif domain == 'google.com':
            referrer_urls.extend([
                'https://www.google.com/search?q=product+search',
                'https://www.google.com/search?q=buy+online',
                'https://www.google.com/search?q=best+products',
                'https://www.google.com'
            ])
        elif domain == 'facebook.com':
            referrer_urls.extend([
                'https://www.facebook.com/ads/123456',
                'https://www.facebook.com/groups/shopping',
                'https://www.facebook.com'
            ])
        else:
            referrer_urls.append(f'https://www.{domain}')
    
    # Device types
    device_types = ['Desktop', 'Mobile', 'Tablet']
    
    # Device models
    device_models = [
        'iPhone', 'Samsung Galaxy', 'Google Pixel', 'iPad', 'MacBook Pro',
        'Dell XPS', 'HP Pavilion', 'Windows PC', 'Android Device', 'Unknown'
    ]
    
    # Browsers
    browsers = ['Chrome', 'Firefox', 'Safari', 'Edge', 'Opera', 'IE', 'Samsung Browser', 'Unknown']
    
    # Operating systems
    operating_systems = ['Windows', 'MacOS', 'iOS', 'Android', 'Linux', 'Chrome OS', 'Unknown']
    
    # Countries
    countries = [
        'USA', 'Canada', 'UK', 'Germany', 'France', 'Australia', 'Japan', 'China',
        'India', 'Brazil', 'Mexico', 'Spain', 'Italy', 'Russia', 'Netherlands',
        'Sweden', 'South Korea', 'Singapore', 'UAE', 'Other'
    ]
    
    # Metrics based on URL type
    def get_metrics_for_url(url):
        # Home and high-traffic pages
        if url == '/' or url in ['/products', '/login', '/signup']:
            pageviews = random.randint(5, 15)
            time_on_page = random.randint(30, 180)
            bounce_rate = random.uniform(0.1, 0.4)
            conversion_rate = random.uniform(0.01, 0.05)
        # Product pages
        elif url.startswith('/products/'):
            pageviews = random.randint(2, 8)
            time_on_page = random.randint(60, 300)
            bounce_rate = random.uniform(0.2, 0.6)
            conversion_rate = random.uniform(0.02, 0.1)
        # Blog pages
        elif url.startswith('/blog'):
            pageviews = random.randint(1, 5)
            time_on_page = random.randint(120, 600)
            bounce_rate = random.uniform(0.4, 0.8)
            conversion_rate = random.uniform(0.005, 0.02)
        # Cart and checkout
        elif url in ['/cart', '/checkout', '/payment', '/confirmation']:
            pageviews = random.randint(1, 3)
            time_on_page = random.randint(60, 240)
            bounce_rate = random.uniform(0.2, 0.5)
            conversion_rate = random.uniform(0.1, 0.4)
        # Other pages
        else:
            pageviews = random.randint(1, 4)
            time_on_page = random.randint(30, 240)
            bounce_rate = random.uniform(0.3, 0.7)
            conversion_rate = random.uniform(0.005, 0.03)
        
        return pageviews, time_on_page, bounce_rate, conversion_rate
    
    # Generate data
    data = {
        'date': dates,
        'hour': [random.randint(0, 23) for _ in range(num_rows)],
        'session_id': [str(uuid.uuid4()) for _ in range(num_rows)],
        'user_id': [],  # Will assign below (some will be anonymous)
        'page_url': [random.choice(page_urls) for _ in range(num_rows)],
        'referrer_url': [random.choice(referrer_urls) for _ in range(num_rows)],
        'device_type': [random.choice(device_types) for _ in range(num_rows)],
        'device_model': [],  # Will assign based on device type
        'browser': [random.choice(browsers) for _ in range(num_rows)],
        'operating_system': [random.choice(operating_systems) for _ in range(num_rows)],
        'country': [random.choice(countries) for _ in range(num_rows)],
        'city': [],  # Will assign based on country
        'pageviews': [],  # Will calculate based on URL
        'time_on_page_seconds': [],  # Will calculate based on URL
        'bounce_rate': [],  # Will calculate based on URL
        'conversion_rate': [],  # Will calculate based on URL
        'is_new_user': [random.choice([True, False]) for _ in range(num_rows)],
        'utm_source': [],  # Will assign based on referrer
        'utm_medium': [],  # Will assign based on referrer
        'utm_campaign': []  # Will assign based on referrer
    }
    
    # Cities by country (simplified)
    cities_by_country = {
        'USA': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego'],
        'Canada': ['Toronto', 'Montreal', 'Vancouver', 'Calgary', 'Ottawa', 'Edmonton'],
        'UK': ['London', 'Manchester', 'Birmingham', 'Glasgow', 'Liverpool', 'Edinburgh'],
        'Germany': ['Berlin', 'Munich', 'Hamburg', 'Cologne', 'Frankfurt', 'Stuttgart'],
        'France': ['Paris', 'Marseille', 'Lyon', 'Toulouse', 'Nice', 'Nantes'],
        'Australia': ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Gold Coast'],
        'Japan': ['Tokyo', 'Osaka', 'Yokohama', 'Nagoya', 'Sapporo', 'Fukuoka'],
        'China': ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Chengdu', 'Tianjin'],
        'India': ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata'],
        'Brazil': ['São Paulo', 'Rio de Janeiro', 'Brasília', 'Salvador', 'Fortaleza', 'Belo Horizonte']
    }
    
    # UTM parameters by referrer type
    utm_by_referrer = {
        'google.com': {
            'source': 'google',
            'medium': 'organic',
            'campaign': ['search', 'brand', 'product']
        },
        'bing.com': {
            'source': 'bing',
            'medium': 'organic',
            'campaign': ['search', 'brand', 'product']
        },
        'facebook.com': {
            'source': 'facebook',
            'medium': ['social', 'cpc'],
            'campaign': ['awareness', 'retargeting', 'conversion', 'promotion']
        },
        'twitter.com': {
            'source': 'twitter',
            'medium': ['social', 'cpc'],
            'campaign': ['awareness', 'engagement', 'promotion']
        },
        'instagram.com': {
            'source': 'instagram',
            'medium': ['social', 'cpc'],
            'campaign': ['awareness', 'product', 'influencer']
        },
        'linkedin.com': {
            'source': 'linkedin',
            'medium': ['social', 'cpc'],
            'campaign': ['b2b', 'career', 'professional']
        },
        'direct': {
            'source': 'direct',
            'medium': 'none',
            'campaign': ['none']
        },
        'email': {
            'source': 'email',
            'medium': 'email',
            'campaign': ['newsletter', 'promotion', 'digest', 'welcome']
        }
    }
    
    # Generate user IDs (some will be anonymous)
    for _ in range(num_rows):
        if random.random() < 0.3:  # 30% chance of anonymous
            data['user_id'].append(None)
        else:
            data['user_id'].append(str(uuid.uuid4()))
    
    # Assign device models based on device type
    for i in range(num_rows):
        device_type = data['device_type'][i]
        
        if device_type == 'Desktop':
            data['device_model'].append(random.choice(['MacBook Pro', 'Dell XPS', 'HP Pavilion', 'Windows PC', 'Unknown']))
        elif device_type == 'Mobile':
            data['device_model'].append(random.choice(['iPhone', 'Samsung Galaxy', 'Google Pixel', 'Android Device', 'Unknown']))
        else:  # Tablet
            data['device_model'].append(random.choice(['iPad', 'Samsung Tab', 'Android Tablet', 'Unknown']))
    
    # Assign cities based on country
    for i in range(num_rows):
        country = data['country'][i]
        
        if country in cities_by_country:
            data['city'].append(random.choice(cities_by_country[country]))
        else:
            data['city'].append('Unknown')
    
    # Calculate metrics based on page URL
    for i in range(num_rows):
        url = data['page_url'][i]
        pageviews, time_on_page, bounce_rate, conversion_rate = get_metrics_for_url(url)
        
        data['pageviews'].append(pageviews)
        data['time_on_page_seconds'].append(time_on_page)
        data['bounce_rate'].append(bounce_rate)
        data['conversion_rate'].append(conversion_rate)
    
    # Assign UTM parameters based on referrer
    for i in range(num_rows):
        referrer = data['referrer_url'][i]
        
        # Extract domain from referrer URL
        if referrer == 'direct' or referrer == 'email':
            domain = referrer
        else:
            try:
                domain = referrer.split('//')[1].split('/')[0].replace('www.', '')
            except:
                domain = 'unknown'
        
        # Find matching domain in utm_by_referrer
        matching_domain = None
        for key in utm_by_referrer.keys():
            if key in domain:
                matching_domain = key
                break
        
        if matching_domain:
            utm_data = utm_by_referrer[matching_domain]
            
            # Assign source
            data['utm_source'].append(utm_data['source'])
            
            # Assign medium
            if isinstance(utm_data['medium'], list):
                data['utm_medium'].append(random.choice(utm_data['medium']))
            else:
                data['utm_medium'].append(utm_data['medium'])
            
            # Assign campaign
            data['utm_campaign'].append(random.choice(utm_data['campaign']))
        else:
            # Default values for unknown referrers
            data['utm_source'].append('other')
            data['utm_medium'].append('referral')
            data['utm_campaign'].append('none')
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Introduce missing values if requested
    if include_missing:
        # Columns that can have missing values (excluding session_id and date)
        missable_columns = [col for col in df.columns if col not in ['session_id', 'date']]
        
        # Introduce missing values randomly
        for col in missable_columns:
            # Determine how many values to set as missing (between 1% and 5%)
            missing_count = int(num_rows * random.uniform(0.01, 0.05))
            
            # Randomly select indices to set as missing
            missing_indices = random.sample(range(num_rows), missing_count)
            
            # Set values as missing
            df.loc[missing_indices, col] = np.nan
    
    # Introduce errors in the data
    if error_rate > 0:
        # Number of errors to introduce
        num_errors = int(num_rows * error_rate)
        
        for _ in range(num_errors):
            # Select a random row
            row_idx = random.randint(0, num_rows - 1)
            
            # Select a random column (excluding session_id)
            col_name = random.choice([col for col in df.columns if col != 'session_id'])
            
            # Introduce an error based on the column type
            if col_name in ['hour', 'pageviews']:
                # For integer columns, introduce negative values or extremely high values
                error_type = random.choice(['negative', 'high_value', 'string'])
                
                if error_type == 'negative':
                    df.at[row_idx, col_name] = -random.randint(1, 100)
                elif error_type == 'high_value':
                    df.at[row_idx, col_name] = random.randint(1000, 10000)
                else:
                    df.at[row_idx, col_name] = random.choice(['N/A', 'unknown', 'error'])
            
            elif col_name in ['time_on_page_seconds', 'bounce_rate', 'conversion_rate']:
                # For float columns, introduce negative values or extremely high values
                error_type = random.choice(['negative', 'high_value', 'string'])
                
                if error_type == 'negative':
                    df.at[row_idx, col_name] = -random.uniform(1, 100)
                elif error_type == 'high_value':
                    if col_name == 'time_on_page_seconds':
                        df.at[row_idx, col_name] = random.uniform(10000, 100000)
                    else:  # bounce_rate, conversion_rate
                        df.at[row_idx, col_name] = random.uniform(5, 100)
                else:
                    df.at[row_idx, col_name] = random.choice(['N/A', 'unknown', 'error'])
            
            elif col_name == 'date':
                # For date columns, introduce invalid dates or future dates
                error_type = random.choice(['invalid_format', 'future_date', 'ancient_date'])
                
                if error_type == 'invalid_format':
                    df.at[row_idx, col_name] = random.choice(['2022/13/45', '01-01-2022', 'not a date', '2022-1-1'])
                elif error_type == 'future_date':
                    future_date = datetime.now() + timedelta(days=random.randint(1, 1000))
                    df.at[row_idx, col_name] = future_date.strftime('%Y-%m-%d')
                else:
                    ancient_date = datetime(1900, 1, 1) + timedelta(days=random.randint(0, 365 * 50))
                    df.at[row_idx, col_name] = ancient_date.strftime('%Y-%m-%d')
            
            elif col_name == 'device_type':
                # For device_type, introduce invalid types
                df.at[row_idx, col_name] = random.choice(['Smart TV', 'Wearable', 'Game Console', 'unknown', '1'])
    
    # Add duplicate rows (about 2% of data)
    num_duplicates = int(num_rows * 0.02)
    duplicate_indices = random.sample(range(num_rows), num_duplicates)
    
    for idx in duplicate_indices:
        # Add the same row again
        df = pd.concat([df, df.iloc[[idx]]], ignore_index=True)
    
    return df

def save_csv(df, filename):
    """Save DataFrame to CSV file."""
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} rows to {filename}")
    print(f"File size: {os.path.getsize(filename) / (1024 * 1024):.2f} MB")

def main():
    """Main function to generate demo datasets."""
    parser = argparse.ArgumentParser(description='Generate demo CSV files for data analysis.')
    parser.add_argument('--output_dir', type=str, default='demo_data', help='Output directory for CSV files')
    parser.add_argument('--num_files', type=int, default=6, help='Number of CSV files to generate')
    parser.add_argument('--error_rate', type=float, default=0.1, help='Error rate for data (0-1)')
    parser.add_argument('--include_missing', action='store_true', help='Include missing values in the data')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Define dataset generators
    generators = [
        ('users', generate_user_data, 100000),
        ('transactions', generate_transaction_data, 200000),
        ('products', generate_product_data, 10000),
        ('customer_support', generate_customer_support_data, 50000),
        ('website_analytics', generate_website_analytics_data, 150000)
    ]
    
    # Select generators based on num_files
    selected_generators = generators[:min(args.num_files, len(generators))]
    
    # Generate and save datasets
    for name, generator_func, size in selected_generators:
        print(f"Generating {name} dataset...")
        df = generator_func(num_rows=size, error_rate=args.error_rate, include_missing=args.include_missing)
        
        # Save to CSV
        filename = os.path.join(args.output_dir, f"{name}_data.csv")
        save_csv(df, filename)
    
    print(f"Generated {len(selected_generators)} CSV files in {args.output_dir}")

if __name__ == "__main__":
    main()
