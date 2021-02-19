# %%
import pandas as pd
import csv
import itertools
import json
import functools

# %%
with open('sqlmatrix.json', 'r') as config_file:
    config = json.load(config_file)

# %%
prices_upapi = pd.read_json(config['upapi_data_path'])
prices_upapi = prices_upapi.set_index('partNumber')

prices_reapi = pd.read_json(config['reapi_data_path'])

with open(config['gaapi_data_path'], 'r') as f:
    prices_gaapi = json.load(f)
prices_gaapi = pd.json_normalize(data=prices_gaapi, sep='_')
prices_gaapi = prices_gaapi.set_index('partNumber')

with open(config['vmspec_data_path'], 'r') as vmspec_file:
    vmspecs = {x['name'].lower(): x for x in csv.DictReader(vmspec_file, quoting=csv.QUOTE_NONNUMERIC)}

# %%
# Parameters
location = config['arm_location']
ahub_sql_2core_pack = config['ahub_sql_server_annual'] / 12.0
ahub_win_server = config['ahub_win_server_annual'] / 12.0

new_ea_discount = float(config['new_ea_discount']) if config['new_ea_discount'] is not None else None
mi_core_multiplier = float(config['mi_core_multiplier']) if config['mi_core_multiplier'] is not None else None
model_mi = config['model_mi']
model_vm = config['model_vm']


# %%
@functools.lru_cache
def get_price_record(part: str, resv_years: int = 0):
    is_meter_id = len(part) == 36
    is_resv = resv_years != 0
    if is_meter_id:
        records = prices_upapi[prices_upapi.meterId == part]
        if not is_resv:
            records = records[pd.isna(records.reservationTerm)]
        else:
            term = "1 Year" if resv_years == 1 else "3 Years" if resv_years == 3 else ""
            records = records[records.reservationTerm == term]

        if records.shape[0] != 1:
            raise Exception('could not reduce to meter')
        record = records.iloc[0]
    else:
        record = prices_upapi.loc[part]

    unitPrice = record.unitPrice

    if is_resv:
        term = record.reservationTerm
        uom = '1/Month'
        divisor = 12 if term == "1 Year" else 36 if term == "3 Years" else 0
        multiplier = 1.0 / divisor
    else:
        uom = record.unitOfMeasure
        if uom.endswith('/Month'):
            multiplier = 1 
        elif uom == '1 Hour': 
            multiplier = 730
            uom = '1/Month'
        else:
            raise Exception('unknown uom')
            
        
        if new_ea_discount is not None:
            part_has_discount = record.unitPrice != record.retailPrice
            if part_has_discount:
                unitPrice = record.retailPrice * ((100 - new_ea_discount) / 100)
            
    return (part, record.productName, record.skuName, uom, unitPrice * multiplier)


@functools.lru_cache
def get_legacy_price_record(part: str):
    record = prices_gaapi.loc[part]
    uom = record.unitOfMeasure
    if uom == '10 Hours': 
        multiplier = 73
        uom = '1/Month'
    elif uom == '1 Hour': 
        multiplier = 730
        uom = '1/Month'
    else:
        raise Exception('unknown uom')
        
    unitPrice = record.unitPrice
    if new_ea_discount is not None:
        raise Exception("can't adjust discount of legacy record")
            
    return (part, record.meterDetails_meterSubCategory, record.meterDetails_meterName, uom, unitPrice * multiplier)


def get_custom_record(product_name: str, sku_name: str, cost: float):
    return (None, product_name, sku_name, '1/Month', cost)


ahub_sql_record = get_custom_record('AHUB', 'SQL 2-core Pack', cost=ahub_sql_2core_pack)
ahub_win_record = get_custom_record('AHUB', 'Windows Server', cost=ahub_win_server)


def build_model(records):
    df = pd.DataFrame(records, columns=['quantity', 'part', 'product_name', 'sku_name', 'unit_of_measure', 'unit_cost', 'record_type'])
    df = df.assign(total_cost=df.quantity * df.unit_cost)
    df['record_type'] = df.record_type.fillna('other')
    return df


def idivceil(n, d):
    return -(n // -d)


def find_vm_billables(vm_size: str):
    s = vmspecs.get(vm_size.lower())
    bill_sku = s['ParentSize'] if s['ParentSize'] else vm_size
    bill_sku = bill_sku.replace('s_', '_').replace('_DS', '_D')
    cores = int(float(s['vCPUsAvailable'] if s['vCPUsAvailable'] else s['vCPUs']))
    mem = float(s['MemoryGB'])
    return (bill_sku, cores, mem)


def find_vm_meter(vm_size: str, payg: bool, resv_years: int):
    price_type = "Consumption" if resv_years == 0 else "Reservation"
    expect_records = 2 if resv_years == 0 else 1
    parts = prices_upapi[(prices_upapi.armSkuName == vm_size) & (prices_upapi.armRegionName == location) & (prices_upapi.type == price_type) & (prices_upapi.serviceName == 'Virtual Machines') & ~prices_upapi.skuName.str.contains('Low Priority') & ~prices_upapi.skuName.str.endswith(' Spot')]
    if resv_years > 0:
         term = "1 Year" if resv_years == 1 else "3 Years" if resv_years == 3 else ""
         parts = parts[parts.reservationTerm == term]

    if parts.shape[0] != expect_records:
        raise Exception('could not reduce to vm_sku parts')

    if payg and resv_years == 0:
        record = parts[parts.productName.str.endswith('Windows')].iloc[0]
    else:
        record = parts[~parts.productName.str.endswith('Windows')].iloc[0]
    
    return record.name if resv_years == 0 else record.meterId


def find_sql_iaas_payg_part(cores: int):
    parts = prices_gaapi[(prices_gaapi.meterDetails_meterCategory == 'Virtual Machines Licenses') & (prices_gaapi.meterDetails_meterSubCategory == 'SQL Server Enterprise')]
    count_str = str(cores) if cores > 4 else '1-4'
    name = f'{count_str} vCPU VM License'
    parts = parts[parts.meterDetails_meterName == name]
    if parts.shape[0] != 1:
        raise Exception('could not reduce to vm_sku parts')
    record = parts.iloc[0]
    return record.name


def iaas_records(vm_size: str, payg: bool, resv_years: int):
    billable_size, billable_cores, _ = find_vm_billables(vm_size)
    vm_meter = find_vm_meter(billable_size, payg, resv_years)
    records = [
        (2, *get_price_record(vm_meter, resv_years)),
        (2, *get_price_record('AAD-18176')),
        (2, *get_price_record('AAD-18155')),
        (2, *get_price_record('AAD-18154'))
    ]
    if payg:
        records += [
            (1, *get_legacy_price_record(find_sql_iaas_payg_part(billable_cores)), 'sql_ip')
        ]
    else:
        billable_win_cores = 2 * billable_cores
        records += [
            (max(billable_cores, 4) // 2, *ahub_sql_record, 'sql_ip'),
            (idivceil(billable_win_cores, 16), *ahub_win_record)
        ]
        
    return records


def sqlmi_records(cores: int, payg: bool, bc: bool, resv_years: int):
    infra_part = 'AAD-33625' if bc else 'AAD-33326'
    disk_part = 'AAD-33594' if bc else 'AAD-33282'
    payg_license_part = 'AAD-33628' if bc else 'AAD-33315'
    
    compute_price_record = get_price_record(infra_part) if resv_years == 0 else sqlmi_resv_infra_records(bc, resv_years)
    storage_price_record = get_price_record(disk_part)
    records = [
        (cores, *compute_price_record),
        (48 * 32, *storage_price_record)
    ]
    
    bill_cores = cores if mi_core_multiplier is None else cores / mi_core_multiplier
    if payg:
        records += [
            (bill_cores, *get_price_record(payg_license_part), 'sql_ip')
        ]
    else:
        # Apply 4:1 AHUB multiplier if general purpose 
        bill_cores = bill_cores if bc else bill_cores / 4           
        # Divide by 2 to get quantity of 2 core Packs.
        records += [
            (bill_cores / 2, *ahub_sql_record, 'sql_ip'),
        ]
    return records


@functools.lru_cache(maxsize=4)
def sqlmi_resv_infra_records(bc: bool, resv_years: int):
    term = "1 Year" if resv_years == 1 else "3 Years" if resv_years == 3 else ""
    divisor = 12 if resv_years == 1 else 36 if resv_years == 3 else 0
    tier = "BC" if bc else "GP"
    sku = f"SQLMI_{tier}_Compute_Gen5"
    records = prices_reapi[(prices_reapi.serviceName.str.contains("SQL Managed Instance")) \
        & (prices_reapi.type == "Reservation") & (prices_reapi.armSkuName == sku) \
        & (prices_reapi.reservationTerm == term)]
    if records.shape[0] != 1: 
        raise Exception('could not find mi resv price')
    record = records.iloc[0]
    return ("N/A", record.productName, record.skuName, "1/Month", record.unitPrice / divisor)


# %%
models = []
bool_options = [True, False]
resv_options = [0, 1, 3]

# %%
vm_size_options = [
    'Standard_E4-2s_v3', 'Standard_E4s_v3', 
    'Standard_E8-2s_v3', 'Standard_E8-4s_v3', 'Standard_E8s_v3',
    'Standard_E16-4s_v3', 'Standard_E16-8s_v3', 'Standard_E16s_v3',
    'Standard_E32-8s_v3', 'Standard_E32-16s_v3', 'Standard_E32s_v3',
    'Standard_E64-16s_v3', 'Standard_E64-32s_v3', 'Standard_E64s_v3',
    'Standard_M8-2ms', 'Standard_M8-4ms', 'Standard_M8ms',
    'Standard_DS11_v2', 
    'Standard_DS12-2_v2', 'Standard_DS12_v2', 
    'Standard_DS13-2_v2', 'Standard_DS13-4_v2', 'Standard_DS13_v2', 
    'Standard_DS14-4_v2', 'Standard_DS14-8_v2', 'Standard_DS14_v2',
    'Standard_E2as_v4', 'Standard_E4as_v4', 'Standard_E8as_v4', 'Standard_E16as_v4',  'Standard_E32as_v4',  'Standard_E64as_v4'
]

if model_vm:
    models += [{'type': 'VM',
                'name': x[0],
                'cores': find_vm_billables(x[0])[1],
                'mem': find_vm_billables(x[0])[2],
                'license': 'PAYG' if x[1] else 'AHUB', 
                'reservation': '3 Years' if x[2] == 3 else '1 Year' if x[2] == 1 else 'None',
                'df': build_model(iaas_records(*x))} 
               for x in itertools.product(vm_size_options, bool_options, resv_options)]


# %%
mi_core_options = [4,8,16,24,32,40,64]

if model_mi:
    models += [{'type': 'MI-BC' if x[2] else 'MI-GP', 
                'name': f'{x[0]} vCores',
                'cores': x[0],
                'mem': x[0] * 5.1,
                'license': 'PAYG' if x[1] else 'AHUB',
                'reservation': '3 Years' if x[3] == 3 else '1 Year' if x[3] == 1 else 'None',
                'df': build_model(sqlmi_records(*x))} 
               for x in itertools.product(mi_core_options, bool_options, bool_options, resv_options)]


# %%
def aggregate_costs(df):
    grouped = df.groupby('record_type').sum().total_cost
    return pd.Series([df.total_cost.sum(), grouped.sql_ip, grouped.other], index=['total_cost', 'sql_ip_cost', 'other_cost'])

summary = pd.DataFrame.from_records(models)
summary = summary.sort_values(['license','type', 'cores', 'mem'])
summary = summary.assign(mem_ratio=summary.mem/summary.cores)
summary = summary.join(summary.df.apply(aggregate_costs))
summary = summary.drop(columns='df')

# %%
summary.style

# %%
summary.to_csv('sqlmatrix.output.csv')


