use std::collections::BTreeSet;
use std::hint::black_box;
use std::time::{Duration, Instant};

use radiate::prelude::*;

const POPULATION_SIZE: usize = 4_096;
const PARENT_DRAWS: usize = 1_000_000;
const SURVIVORS: usize = 1_024;
const SEED: u64 = 20_260_828;

fn main() {
    let population = population();

    let (custom_parents, custom_parent_elapsed) = timed(|| custom_parent_indices(PARENT_DRAWS));
    let (radiate_parents, radiate_parent_elapsed) = timed(|| radiate_tournament(&population));
    let radiate_parents_repeat = radiate_tournament(&population);

    let (scalar_survivors, scalar_elapsed) = timed(scalar_top_k);
    let (nsga2_survivors, nsga2_elapsed) = timed(|| radiate_nsga2(&population));
    let nsga2_repeat = radiate_nsga2(&population);
    let (nsga3_survivors, nsga3_elapsed) = timed(|| radiate_nsga3(&population));
    let nsga3_repeat = radiate_nsga3(&population);

    assert_eq!(radiate_parents, radiate_parents_repeat);
    assert_eq!(nsga2_survivors, nsga2_repeat);
    assert_eq!(nsga3_survivors, nsga3_repeat);

    println!(
        concat!(
            "{{\n",
            "  \"schema\": 1,\n",
            "  \"radiate_version\": \"1.3.0\",\n",
            "  \"seed\": {SEED},\n",
            "  \"population_size\": {POPULATION_SIZE},\n",
            "  \"parent_draws\": {PARENT_DRAWS},\n",
            "  \"survivors\": {SURVIVORS},\n",
            "  \"custom_parent_index\": {custom_parent},\n",
            "  \"radiate_tournament_k3\": {radiate_parent},\n",
            "  \"scalar_top_k_reference\": {scalar},\n",
            "  \"radiate_nsga2\": {nsga2},\n",
            "  \"radiate_nsga3_p12\": {nsga3}\n",
            "}}"
        ),
        SEED = SEED,
        POPULATION_SIZE = POPULATION_SIZE,
        PARENT_DRAWS = PARENT_DRAWS,
        SURVIVORS = SURVIVORS,
        custom_parent = metrics(&custom_parents, custom_parent_elapsed),
        radiate_parent = metrics(&radiate_parents, radiate_parent_elapsed),
        scalar = metrics(&scalar_survivors, scalar_elapsed),
        nsga2 = metrics(&nsga2_survivors, nsga2_elapsed),
        nsga3 = metrics(&nsga3_survivors, nsga3_elapsed),
    );
}

fn population() -> Vec<Phenotype<IntChromosome<i32>>> {
    (0..POPULATION_SIZE)
        .map(|index| {
            let chromosome = IntChromosome::from(vec![i32::try_from(index).unwrap()]);
            let mut member = Phenotype::from((vec![chromosome], 0));
            member.set_score(Some(
                vec![
                    unit_score(index, 17),
                    unit_score(index, 43),
                    unit_score(index, 97),
                ]
                .into(),
            ));
            member
        })
        .collect()
}

fn unit_score(index: usize, multiplier: usize) -> f32 {
    let scrambled = index.wrapping_mul(multiplier) % POPULATION_SIZE;
    1.0 - scrambled as f32 / POPULATION_SIZE as f32
}

fn custom_parent_indices(count: usize) -> Vec<usize> {
    (0..count)
        .map(|slot| {
            slot.wrapping_mul(17)
                .wrapping_add(1)
                .wrapping_rem(POPULATION_SIZE)
        })
        .collect()
}

fn radiate_tournament(population: &[Phenotype<IntChromosome<i32>>]) -> Vec<usize> {
    random_provider::scoped_seed(SEED, || {
        TournamentSelector::new(3).select(
            population,
            &Objective::Single(Optimize::Maximize),
            PARENT_DRAWS,
        )
    })
}

fn scalar_top_k() -> Vec<usize> {
    let mut indices = (0..POPULATION_SIZE).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        let left_score =
            unit_score(*left, 17) + 0.7 * unit_score(*left, 43) + 0.3 * unit_score(*left, 97);
        let right_score =
            unit_score(*right, 17) + 0.7 * unit_score(*right, 43) + 0.3 * unit_score(*right, 97);
        right_score
            .total_cmp(&left_score)
            .then_with(|| left.cmp(right))
    });
    indices.truncate(SURVIVORS);
    indices
}

fn radiate_nsga2(population: &[Phenotype<IntChromosome<i32>>]) -> Vec<usize> {
    NSGA2Selector::new().select(population, &multi_objective(), SURVIVORS)
}

fn radiate_nsga3(population: &[Phenotype<IntChromosome<i32>>]) -> Vec<usize> {
    NSGA3Selector::new(12).select(population, &multi_objective(), SURVIVORS)
}

fn multi_objective() -> Objective {
    Objective::Multi(vec![Optimize::Maximize; 3])
}

fn timed<T>(run: impl FnOnce() -> T) -> (T, Duration) {
    let started = Instant::now();
    let result = black_box(run());
    (result, started.elapsed())
}

fn metrics(indices: &[usize], elapsed: Duration) -> String {
    let checksum = indices
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, index| {
            hash.wrapping_mul(0x0000_0100_0000_01b3) ^ u64::try_from(*index).unwrap()
        });
    let unique = indices.iter().copied().collect::<BTreeSet<_>>().len();
    let mean_index = indices.iter().map(|index| *index as f64).sum::<f64>() / indices.len() as f64;
    format!(
        concat!(
            "{{\"elapsed_us\":{},\"checksum_fnv64\":\"{:016x}\",",
            "\"unique_indices\":{},\"mean_index\":{:.3}}}"
        ),
        elapsed.as_micros(),
        checksum,
        unique,
        mean_index,
    )
}
