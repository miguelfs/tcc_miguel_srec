select t.session_id,
       s.user_id                                                            as owner_id,
       t.user_id,
       min(UNIX_SECONDS(TIMESTAMP(t.created_at)))                 as created_at,
       case when t.parent_id = '' then 'create_track' else 'fork_track' end as action
from ${project_id}.${db_name}.tracks t
         inner join ${project_id}.${db_name}.sessions s on t.session_id = s.id
where t.created_at > '2023-01-01'
group by  t.session_id, s.user_id, t.user_id, action
order by created_at desc
